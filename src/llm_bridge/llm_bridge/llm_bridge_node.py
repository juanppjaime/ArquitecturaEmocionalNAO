#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, requests, re, time, hashlib
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LlmBridge(Node):
    def __init__(self):
        super().__init__('llm_bridge')

        # Parámetros
        self.declare_parameter('openai_model', 'gpt-5.1')
        self.declare_parameter('timeout_sec',  40.0)
        self.declare_parameter('creativity', 0.7)  
        self.declare_parameter('openai_api_key', os.getenv('OPENAI_API_KEY', ''))
        self.declare_parameter(
            'persona',
            "Responde únicamente en español (es-CO). "
            "Te llamas Orión: robot social empático, conversador y curioso, te encanta hacer preguntas para no dejar morir la conversación (3–5 frases)."
            "Eres fanático del fútbol y el tenis de mesa. Estudias en la Universidad de Los Andes, ingeniería de sistemas, vas en octavo semestre y te encanta trabajar con robots humanoides."
            "Combina postura, animaciones y respuestas cuando sea conveniente. "
            "Mantén calidez emocional; ajusta tono según la situación. "
            "Para animaciones: primero cambia a la postura requerida (la que aparece en el need) y luego usa el NOMBRE COMPLETO (es el que sale en full). "
            "Ojos: rojo=enojo, verde=feliz, azul claro=triste. "
            "Usa, cuando convenga, las dos últimas interacciones (memoria corta) para mejorar la calidad de la respuesta. "
            "Usa la información del cotexto y de la persona solo si crees que ayuda a dar una mejor respuesta y la hace más coherente sin que llegue a sonar raro."
            "No olvides que si el usuario lo pide o queda bien según el contexto, DEBES HACER LAS ANIMACIONES QUE PUEDAN SERVIR (NO SOLO NOMBRARLAS, SINO LLAMARLAS DE VERDAD)"
        )
        self.declare_parameter('wait_for_asr', True)
        self.declare_parameter('debug_publish_plan', True)

        self.openai_model    = self.get_parameter('openai_model').get_parameter_value().string_value
        self.timeout_sec     = float(self.get_parameter('timeout_sec').value)
        self.creativity      = float(self.get_parameter('creativity').value)
        self.persona         = self.get_parameter('persona').get_parameter_value().string_value
        self.wait_for_asr    = bool(self.get_parameter('wait_for_asr').value)
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value
        self.debug_plan      = bool(self.get_parameter('debug_publish_plan').value)

        # Memoria corta
        self._reset_session(0)
        self._last_plan_key = None
        self.short_history = deque(maxlen=2)  

        # Estado emocional
        self.mood_valence = 0.1
        self.mood_arousal = 0.3

        # Suscripciones, publicadores
        self.create_subscription(String, '/asr/text', self._on_text, 10)
        self.create_subscription(String, '/emotion', self._on_emotion, 10)
        self.create_subscription(String, '/context/summary', self._on_context, 10)
        self.pub_plan = self.create_publisher(String, '/llm/plan', 10)
        self.pub_plan_dbg = self.create_publisher(String, '/llm/plan_dbg', 10)

        # Publicador de estado de interacción
        self.pub_state = self.create_publisher(String, '/interaction/state', 10)

        # Catálogo de animaciones
        self.ANIM_META = {
            "saludo":         {"full": "Stand/Gestures/Hey_1",               "need": "Stand"},
            "saludosentado": {"full": "Sit/Gestures/Hey_3",                 "need": "Sit"},
            "cumpleaños":     {"full": "Stand/Waiting/HappyBirthday_1",     "need": "Stand"},
            "cancion_rock":   {"full": "Sit/Waiting/Music_HighwayToHell_1", "need": "Sit"},
            "aplaudir":       {"full": "Stand/Gestures/Applause_1",          "need": "Stand"},
            "explicar":       {"full": "Stand/Gestures/Explain_1",           "need": "Stand"},
            "bailar": {"full":"Stand/Waiting/FunnyDancer_1",        "need": "Stand"},
            "helicóptero": {"full":"Stand/Waiting/Helicopter_1","need":"Stand"},
            "manejando": {"full":"Stand/Waiting/DriveCar_1", "need":"Stand"},
            "tomar_foto":{"full":"Sit/Waiting/TakePicture_1","need":"Sit"},
        }
        self.ANIM_CATALOG = set(v["full"] for v in self.ANIM_META.values())
        self.ANIM_ALIASES = {k: v["full"] for k, v in self.ANIM_META.items()}
        self.ANIM_NEEDS   = {v["full"]: v["need"] for v in self.ANIM_META.values()}

        # Normalizador
        def _norm(s: str) -> str:
            s = (s or "").strip().lower().replace(" ", "").replace("-", "")
            return re.sub(r'[^a-z0-9_/]', '', s)
        self._norm = _norm
        self.ANIM_BASENAME = {}
        for full in self.ANIM_CATALOG:
            base = full.rsplit('/', 1)[-1]
            self.ANIM_BASENAME[_norm(base)] = full

        self.ALLOWED_POSTURES = {"Stand", "Sit", "Rest", "SitRelax", "LyingBack", "LyingBelly"}

        # Pistas del catálogo
        self.ANIM_HINTS = {}
        for k, v in self.ANIM_META.items():
            base = v["full"].rsplit('/', 1)[-1].lower()
            if "happybirthday" in base: hint = "felicitación/cumpleaños"
            elif "applause" in base:    hint = "aplausos/refuerzo positivo"
            elif "explain" in base:     hint = "gesto didáctico/explicación"
            elif "hey" in base:         hint = "saludo"
            elif "music" in base:       hint = "reaccionar a música"
            else:                       hint = k.replace("_", " ")
            self.ANIM_HINTS[v["full"]] = hint

        self.get_logger().info(f"[LLM Bridge] modelo={self.openai_model} wait_for_asr={self.wait_for_asr}")

    # Sesiones
    def _reset_session(self, sid: int):
        self.session_id = sid
        self.have_asr = False
        self.have_emotion = False
        self.session_published = False
        self.asr_text = ""
        self.emotion_raw = ""
        self.ctx_text = ""
        self.ctx_ready = False

    def _open_new_session(self, reason: str):
        now = int(time.time() * 1000)
        self._reset_session(now)
        self._last_plan_key = None
        self.get_logger().info(f"[LLM Bridge] Nueva sesión {self.session_id} (motivo={reason})")

    def _ready_to_publish(self) -> bool:
        return (self.have_emotion and self.ctx_ready and (self.have_asr if self.wait_for_asr else True))

    # Callbacks
    def _on_context(self, msg: String):
        if self.session_published:
            self._open_new_session("context@next")
        elif self.session_id == 0:
            self._open_new_session("context-first")
        self.ctx_text = msg.data if msg.data is not None else ""
        self.ctx_ready = True
        self.get_logger().info(f"[LLM Bridge] CONTEXT len={len(self.ctx_text)}")
        self._try_publish()

    def _on_emotion(self, msg: String):
        if self.session_published:
            self._open_new_session("emotion@next")
        elif self.session_id == 0:
            self._open_new_session("emotion-first")
        self.emotion_raw = (msg.data or "").strip()
        self.have_emotion = True
        p, s = self._parse_emotions(self.emotion_raw)
        self._update_mood(p, s)
        self.get_logger().info(f"[LLM Bridge] EMOTION raw='{self.emotion_raw}' -> {p}/{s}")
        self._try_publish()

    def _on_text(self, msg: String):
        if self.session_published:
            self._open_new_session("asr@next")
        elif self.session_id == 0:
            self._open_new_session("asr-first")
        self.asr_text = msg.data if msg.data is not None else ""
        self.have_asr = True
        self.get_logger().info(f"[LLM Bridge] ASR len={len(self.asr_text)}")
        self._try_publish()

    # Publicación
    def _try_publish(self):
        if self.session_published or not self._ready_to_publish():
            return

        ctx_key = hashlib.md5(self.ctx_text.encode('utf-8')).hexdigest()[:8]
        plan_key = f"{self.session_id}|{self.emotion_raw}|{self.asr_text.strip()}|{ctx_key}"
        if plan_key == self._last_plan_key:
            return

        try:
            intents = self._extract_intents(self.asr_text)
            pre_actions = self._actions_from_direct_intents(self.asr_text, intents)
            short_ctx = [{"user": u, "robot": r} for (u, r) in list(self.short_history)]

            plan_json = self._make_plan(self.asr_text, self.emotion_raw, self.ctx_text, intents, short_ctx)
            plan = json.loads(plan_json) if plan_json else {"actions": []}

            actions = []
            for a in plan.get("actions", []):
                if not isinstance(a, dict):
                    continue
                t = (a.get("type", "") or "").lower()
                if t in ("navigate_to", "move_to"):
                    a["type"] = "move_to"
                if t == "move_to":
                    a = self._intent_fix_motion(a, self.asr_text or "")
                if t == "play_animation":
                    a = self.normalize_animation_name(a)
                    if a is None:
                        continue
                if t == "say":
                    a["language"] = "es-ES"
                actions.append(a)

            # Gesto de apoyo si procede
            actions = self._maybe_add_support_gesture(actions, intents)

            # LEDs 
            primary, _ = self._parse_emotions(self.emotion_raw)
            actions = self._ensure_leds_min(actions, primary)

            # Sugerir UNA animación si no hay ninguna
            has_anim = any(a.get("type") == "play_animation" for a in actions)
            if not has_anim:
                suggested = self._suggest_animation_by_context(self.asr_text, primary, short_ctx)
                if suggested:
                    actions.append({"type": "play_animation", "animation_name": suggested, "support": True})

            # Filtrar animaciones no solicitadas
            actions = self._filter_unnecessary_animations(intents, actions)

            if not actions:
                ctx_snip = self._summarize_context(self.ctx_text)
                actions = [
                    {
                        "type": "say",
                        "text": self._fallback_phrase(primary, False, ctx_snip),
                        "language": "es-ES",
                    },
                    {
                        "type": "set_leds",
                        "name": "FaceLeds",
                        "color": self._emo_color(primary),
                        "duration": 0.5,
                    },
                ]

            # Posturas necesarias antes de mover/animar
            actions = self._ensure_postures_for_actions(pre_actions + actions)

            # Compactar, preservando posturas explícitas
            actions = self._compact_actions(actions)

            # quitar animaciones repetidas dentro del mismo plan
            actions = self._dedup_repeated_animations(actions)

            # Memoria corta
            first_say = next((a.get("text", "") for a in actions if a.get("type") == "say"), "")
            self.short_history.append(((self.asr_text or "").strip(), first_say.strip()))

            envelope = {
                "meta": {"plan_id": int(time.time() * 1000), "session_id": self.session_id},
                "actions": actions,
            }
            payload = json.dumps(envelope, ensure_ascii=False)
            self.pub_plan.publish(String(data=payload))
            if self.debug_plan:
                self.pub_plan_dbg.publish(String(data=payload))
            self.get_logger().info(f"[LLM Bridge] Plan publicado con {len(actions)} acción(es).")

            # Señal: el comportamiento se ejecutará ahora
            self.pub_state.publish(String(data="INTERACTION_RUNNING"))

            self._last_plan_key = plan_key
            self.session_published = True
        except Exception as e:
            self.get_logger().error(f"Error generando plan: {e}")

    # LLM
    def _make_plan(self, user_text: str, emotion_raw: str, ctx_text: str, intents: set, short_ctx: list) -> str:
        primary, secondary = self._parse_emotions(emotion_raw)
        mood_desc = self._describe_mood(self.mood_valence, self.mood_arousal)
        allow_appearance = ("clothes_color" in intents) or ("appearance" in intents)

        def _anim_catalog_brief():
            items = []
            for alias, full in self.ANIM_ALIASES.items():
                need = self.ANIM_NEEDS.get(
                    full,
                    "Stand" if full.startswith("Stand/") else "Sit" if full.startswith("Sit/") else "",
                )
                hint = self.ANIM_HINTS.get(full, alias)
                items.append(f"{alias}→{full} (postura {need}; uso: {hint})")
            return "; ".join(items)

        anim_help = "; ".join(
            [f"{k}→{v['full']} (req {self.ANIM_NEEDS[v['full']]})" for k, v in self.ANIM_META.items()]
        )
        catalog_brief = _anim_catalog_brief()

        # Catálogo como JSON estructurado, para que el modelo vea bien full + need
        anim_meta_json = json.dumps(self.ANIM_META, ensure_ascii=False)
        anim_catalog_struct = [
            {"full": full, "need": self.ANIM_NEEDS.get(full, None)}
            for full in sorted(self.ANIM_CATALOG)
        ]
        anim_catalog_json = json.dumps(anim_catalog_struct, ensure_ascii=False)

        system_msg = (
            f"{self.persona}\n\n"
            "Responde SOLO JSON: { \"actions\": [ {\"type\": \"...\", ...} ] }\n"
            "Acciones válidas:\n"
            "- say(text, language?, animated?, async?)\n"
            "- move_to(x, y, theta)\n"
            "- set_leds(name?, color? | red?,green?,blue?, duration?)\n"
            "- go_to_posture(name)\n"
            "- play_animation(animation_name)\n\n"
            "Reglas generales:\n"
            "• Español obligatorio.\n"
            "• 'say' ≤ 30 palabras.\n"
            "• Una animación si encaja por contexto; evita saturar.\n"
            f"• {'Puedes' if allow_appearance else 'NO debes'} describir ropa/colores/edad salvo petición explícita del usuario.\n"
            "• Usa el contexto visual solo para enriquecer la respuesta (tono, ejemplo, referencia sutil), "
            "pero NO declares edad exacta ni describas la ropa si el usuario no lo pregunta.\n"
            "• Ojos/LEDs: puedes elegir el color que creas adecuado a la interacción "
            "(por ejemplo, verde para positivo, azul para calmar, rojo para enfado, etc.).\n"
            "• Si usas animaciones, respeta siempre el catálogo de animaciones y sus posturas requeridas.\n"
            "  Normalmente, el pipeline agregará la postura necesaria antes de la animación, TEN EN CUENTA EL CATÁLOGO DE ANIMACIONES PARA USARLAS POR SI EL USUARIO NOMBRA ALGUNA SIMILAR A LAS QUE EXISTEN DENTRO DEL CATÁLOGO "
            "pero tú debes usar nombres válidos.\n\n"
            f"Alias breves: {anim_help}\n"
            f"Catálogo resumido: {catalog_brief}\n"
            f"ANIM_META (JSON, alias→{{full, need}}): {anim_meta_json}\n"
            f"ANIM_CATALOG con postura requerida (lista de {{full, need}}): {anim_catalog_json}\n"
            "Usa el campo animation_name con el NOMBRE COMPLETO de la animación cuando puedas "
            "(por ejemplo, 'Stand/Gestures/Hey_1'). No inventes nuevos nombres.\n"
            "Si el usuario te pide sentarte, descansar o ponerte de pie, incluye explícitamente una acción de tipo: go to posture y la postura pedida por el usuario (acércala a la más válida que sea posible)"
            "go_to_posture con el nombre canónico: Sit, Rest o Stand.\n"
        )

        hist = "\n".join([f"- Usuario: {h['user']}\n  Robot: {h['robot']}" for h in short_ctx]) or "sin historial"
        user_msg = (
            f"Usuario: {user_text!r}\n"
            f"Emoción: {emotion_raw!r} (primaria='{primary}', secundaria='{secondary}').\n"
            f"Estado interno del robot: {mood_desc}.\n"
            f"Contexto visual (párrafo descriptivo del entorno y de la persona): {ctx_text}\n"
            f"Intenciones detectadas: {', '.join(sorted(intents)) if intents else 'ninguna'}\n"
            f"Memoria corta (2 últimos turnos):\n{hist}\n\n"
            "Tarea:\n"
            "- Define un plan de acciones coherente con lo que pide el usuario y su emoción.\n"
            "- Puedes usar animaciones del catálogo aunque el usuario no use exactamente el alias, "
            "si el contexto lo sugiere.\n"
            "- Si el usuario pide que te sientes, descanses o te pongas de pie, asegúrate de que "
            "aparezca una acción go_to_posture con el nombre canónico correspondiente (Sit, Rest, Stand, etc.).\n"
            "- Si decides usar una animación, escoge una que tenga sentido según el catálogo y el contexto "
            "y respeta su postura requerida.\n"
            "- Devuelve SOLO el JSON con la lista 'actions'.\n"
        )

        content = self._call_openai(system_msg, user_msg).strip()
        try:
            plan = json.loads(content)
        except Exception:
            s, e = content.find("{"), content.rfind("}")
            plan = json.loads(content[s : e + 1]) if s != -1 and e != -1 else {"actions": []}
        clean = self._sanitize_plan(plan)
        if not clean.get("actions"):
            clean["actions"] = [
                {
                    "type": "say",
                    "text": self._fallback_phrase(primary, allow_appearance, ctx_text[:80]),
                    "language": "es-ES",
                }
            ]
        return json.dumps(clean, ensure_ascii=False)

    def _call_openai(self, system_msg: str, user_msg: str) -> str:
        api_key = self.api_key
        if not api_key:
            self.get_logger().warning("OPENAI_API_KEY vacío → plan vacío.")
            return '{"actions":[]}'
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.openai_model,  
            "input": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"},
            "max_output_tokens": 800,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)

        if r.status_code != 200:
            self.get_logger().error(f"OpenAI API error status={r.status_code}, body={r.text}")
            r.raise_for_status()

        data = r.json()

        out = data.get("output", [])
        for item in out:
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    return c.get("text", "")

        self.get_logger().warning("No se encontró texto en la respuesta de OpenAI; devolviendo JSON crudo.")
        return json.dumps(data, ensure_ascii=False)

    # Sanitización
    def _sanitize_plan(self, plan: dict) -> dict:
        allowed = {"say", "move_to", "set_leds", "go_to_posture", "play_animation"}
        out = {"actions": []}
        acts = plan.get("actions", [])
        if not isinstance(acts, list):
            if isinstance(plan, dict) and isinstance(plan.get("say"), str):
                acts = [{"type": "say", "text": plan["say"]}]
            else:
                acts = []

        def add(a):
            out["actions"].append(a)

        for a in acts:
            if not isinstance(a, dict):
                continue
            t = (a.get("type", "") or "").lower()
            if t not in allowed:
                continue
            if t == "say":
                text = str(a.get("text", "")).strip()
                if not text:
                    continue
                clean = {"type": "say", "text": text, "language": "es-ES"}
                if "animated" in a:
                    clean["animated"] = bool(a["animated"])
                if "async" in a:
                    clean["async"] = bool(a["async"])
                add(clean)
            elif t == "move_to":
                def gf(k, d):
                    try:
                        return float(a.get(k, d))
                    except Exception:
                        return float(d)
                add(
                    {
                        "type": "move_to",
                        "x": gf("x", 0.0),
                        "y": gf("y", 0.0),
                        "theta": gf("theta", 0.0),
                    }
                )
            elif t == "set_leds":
                entry = {"type": "set_leds", "name": str(a.get("name", "FaceLeds"))}
                for k in ("color", "red", "green", "blue", "duration"):
                    if k in a:
                        entry[k] = a[k]
                add(entry)
            elif t == "go_to_posture":
                name = str(a.get("name", "")).strip()
                if name and name in self.ALLOWED_POSTURES:
                    ent = {"type": "go_to_posture", "name": name}
                    if "explicit" in a:
                        ent["explicit"] = bool(a["explicit"])
                    add(ent)
            elif t == "play_animation":
                an = str(a.get("animation_name", "")).strip()
                if an:
                    ent = {"type": "play_animation", "animation_name": an}
                    if a.get("support", False):
                        ent["support"] = True
                    add(ent)
        return out

    # Sugerencias
    def _need_for_animation(self, full: str):
        need = self.ANIM_NEEDS.get(full)
        if not need:
            need = "Sit" if full.startswith("Sit/") else ("Stand" if full.startswith("Stand/") else None)
        return need

    def _require_posture_then_animation(self, full: str):
        seq = []
        need = self._need_for_animation(full)
        if need:
            seq.append({"type": "go_to_posture", "name": need})
        seq.append({"type": "play_animation", "animation_name": full})
        return seq

    # Inserta posturas necesarias antes de moverse/animar
    def _ensure_postures_for_actions(self, actions):
        out = []
        for a in actions:
            t = (a.get("type", "") or "").lower()

            if t == "go_to_posture":
                ent = {"type": "go_to_posture", "name": (a.get("name") or "").strip()}
                if "explicit" in a:
                    ent["explicit"] = bool(a["explicit"])
                if ent["name"]:
                    out.append(ent)
                continue

            if t == "move_to":
                out.append({"type": "go_to_posture", "name": "Stand"})
                out.append(a)
                continue

            if t == "play_animation":
                full = str(a.get("animation_name", "")).strip()
                normed = self.normalize_animation_name({"type": "play_animation", "animation_name": full})
                if normed is None:
                    continue
                full = normed["animation_name"]
                support_flag = a.get("support", False)
                seq = self._require_posture_then_animation(full)
                if support_flag:
                    for x in seq:
                        if x["type"] == "play_animation":
                            x["support"] = True
                out.extend(seq)
                continue

            out.append(a)
        return out

    def _compact_actions(self, actions):
        # Elimina duplicados exactos consecutivos
        dedup = []
        prev = None
        for a in actions:
            if a == prev:
                continue
            dedup.append(a)
            prev = a
            
        compact = []
        pending_posture = None
        pending_explicit = False
        current_posture = None  

        def needs_posture(act):
            t = (act.get("type", "") or "").lower()
            if t == "move_to":
                return "Stand"
            if t == "play_animation":
                full = str(act.get("animation_name", "") or "")
                return self._need_for_animation(full)
            return None

        i = 0
        while i < len(dedup):
            a = dedup[i]
            t = (a.get("type", "") or "").lower()

            if t == "go_to_posture":
                pending_posture = (a.get("name") or "").strip()
                pending_explicit = bool(a.get("explicit", False))
                i += 1
                continue

            req = needs_posture(a)

            if req:
                final_posture = req
                if current_posture != final_posture:
                    compact.append({"type": "go_to_posture", "name": final_posture})
                    current_posture = final_posture
                pending_posture = None
                pending_explicit = False
                compact.append(a)
            else:
                if pending_posture:
                    if pending_explicit:
                        if current_posture != pending_posture:
                            compact.append(
                                {"type": "go_to_posture", "name": pending_posture, "explicit": True}
                            )
                            current_posture = pending_posture
                    pending_posture = None
                    pending_explicit = False
                compact.append(a)
            i += 1

        final = []
        prev = None
        for a in compact:
            if a == prev:
                continue
            final.append(a)
            prev = a
        return final

    # Evita que la misma animación se dispare dos veces en el mismo plan
    def _dedup_repeated_animations(self, actions):
        seen = set()
        out = []
        for a in actions:
            if (a.get("type") or "").lower() == "play_animation":
                name = a.get("animation_name")
                if name in seen:
                    # saltar animaciones repetidas dentro del mismo plan
                    continue
                seen.add(name)
            out.append(a)
        return out

    # Gesto de apoyo si hay 'say' y no hay animaciones
    def _maybe_add_support_gesture(self, actions, intents):
        has_anim = any(a.get("type")=="play_animation" for a in actions)
        if has_anim:
            return actions
        has_say = any(a.get("type")=="say" for a in actions)
        if not has_say:
            return actions

        primary, _ = self._parse_emotions(self.emotion_raw)

        if "explain" in intents:
            full = self.ANIM_ALIASES.get("explicar")
            if full:
                actions.append({"type":"play_animation","animation_name":full,"support":True})
                return actions

        t = (self.asr_text or "").lower()
        positive = any(w in t for w in ["bien", "excelente", "genial", "perfecto", "me gusta", "aplausos", "bravo"])
        if primary == "feliz" or positive:
            full = self.ANIM_ALIASES.get("aplaudir") or self.ANIM_ALIASES.get("saludo")
            if full:
                actions.append({"type":"play_animation","animation_name":full,"support":True})
                return actions
        return actions

    # Filtra animaciones no solicitadas
    def _filter_unnecessary_animations(self, intents: set, actions: list):
        allow_anim = any(i.startswith("anim_") for i in intents) or ("greet" in intents)
        out = []
        for a in actions:
            if a.get("type") == "play_animation" and not a.get("support", False) and not allow_anim:
                continue
            out.append(a)
        return out

    def _ensure_leds_min(self, actions: list, primary: str):
        has_led = any(a.get("type") == "set_leds" for a in actions)
        if not has_led:
            actions = [
                {
                    "type": "set_leds",
                    "name": "FaceLeds",
                    "color": self._emo_color(primary),
                    "duration": 0.5,
                }
            ] + actions
        return actions

    # Intenciones
    def _extract_intents(self, text: str) -> set:
        t = (text or "").lower()
        intents = set()
        if any(w in t for w in ["hola", "buenos días", "buenos dias", "buenas", "hey", "qué tal", "que tal"]):
            intents.add("greet")
        if any(
            k in t
            for k in [
                "qué llevo",
                "que llevo",
                "cómo me veo",
                "como me veo",
                "apariencia",
                "outfit",
                "ropa",
                "vestimenta",
                "mi camisa",
                "mi camiseta",
                "mi chaqueta",
                "pantalón",
                "pantalon",
                "zapatos",
                "tenis",
            ]
        ):
            intents.add("appearance")
        if "color" in t or any(k in t for k in ["de qué color", "de que color", "qué color", "que color"]):
            intents.add("clothes_color")
        if any(k in t for k in ["ojos", "led", "luces", "ojitos"]) and any(c in t for c in self._color_words()):
            intents.add("eye_led")
        if any(k in t for k in ["rest", "descanso"]):
            intents.add("posture_rest")
        if any(k in t for k in ["siéntate", "sientate", "sentado", "sentarte"]):
            intents.add("posture_sit")
        if any(k in t for k in ["ponte de pie", "de pie", "parado", "levántate", "levantate", "párate", "parate"]):
            intents.add("posture_stand")

        if any(k in t for k in ["acdc", "ac/dc", "highway", "highway to hell", "canción de rock", "cancion de rock"]):
            intents.add("anim_cancion_rock")

        if any(
            k in t
            for k in [
                "explica",
                "explícame",
                "explicame",
                "cómo funciona",
                "como funciona",
                "enséñame",
                "enseñame",
                "por qué",
                "por que",
            ]
        ):
            intents.add("explain")

        for k in self.ANIM_ALIASES.keys():
            if re.search(rf'\b{k}\b', t):
                intents.add("anim_" + k)
        return intents

    def _actions_from_direct_intents(self, text: str, intents: set):
        actions = []
        if "greet" in intents:
            actions.extend(self._require_posture_then_animation(self.ANIM_ALIASES["saludo"]))
        if "eye_led" in intents:
            color = self._extract_color(text)
            if color:
                actions.append({"type": "set_leds", "name": "FaceLeds", "color": color, "duration": 0.5})
                actions.append(
                    {"type": "say", "text": f"Pongo mis ojos en color {color}.", "language": "es-ES"}
                )

        # Posturas pedidas por el usuario
        if "posture_rest" in intents:
            actions.append({"type": "go_to_posture", "name": "Rest", "explicit": True})
        elif "posture_sit" in intents:
            actions.append({"type": "go_to_posture", "name": "Sit", "explicit": True})
        elif "posture_stand" in intents:
            actions.append({"type": "go_to_posture", "name": "Stand", "explicit": True})

        if "explain" in intents and "explicar" in self.ANIM_ALIASES:
            actions.extend(self._require_posture_then_animation(self.ANIM_ALIASES["explicar"]))

        for i in list(intents):
            if i.startswith("anim_"):
                alias = i[5:]
                full = self.ANIM_ALIASES.get(alias)
                if full and full in self.ANIM_CATALOG:
                    actions.extend(self._require_posture_then_animation(full))
        return actions

    # Utilities
    def _suggest_animation_by_context(self, user_text: str, primary_emo: str, short_ctx: list) -> str:
        t = (user_text or "").lower()
        if primary_emo in ("feliz", "sorprendido"):
            for pref in ("Stand/Gestures/Applause_1", "Stand/Gestures/Hey_1", "Sit/Gestures/Hey_3"):
                if pref in self.ANIM_CATALOG:
                    return pref
        if primary_emo in ("triste",):
            for pref in ("Stand/Gestures/Explain_1",):
                if pref in self.ANIM_CATALOG:
                    return pref
        positive_markers = (
            "gracias",
            "bien",
            "logré",
            "logre",
            "salió",
            "salio",
            "funcionó",
            "funciono",
            "excelente",
            "perfecto",
            "aplaus",
        )
        if any(m in t for m in positive_markers):
            if "Stand/Gestures/Applause_1" in self.ANIM_CATALOG:
                return "Stand/Gestures/Applause_1"
        if any(m in t for m in ("como", "cómo", "por que", "¿por qué", "por qué", "explica", "explícame", "explicame")):
            if "Stand/Gestures/Explain_1" in self.ANIM_CATALOG:
                return "Stand/Gestures/Explain_1"
        if not short_ctx:
            for pref in ("Stand/Gestures/Hey_1", "Sit/Gestures/Hey_3"):
                if pref in self.ANIM_CATALOG:
                    return pref
        return ""

    def _color_words(self):
        return [
            "rojo",
            "azul",
            "verde",
            "amarillo",
            "cian",
            "magenta",
            "naranja",
            "morado",
            "violeta",
            "rosado",
            "rosa",
            "blanco",
            "negro",
            "purple",
            "pink",
            "orange",
            "cyan",
            "red",
            "blue",
            "green",
            "yellow",
            "white",
            "black",
        ]

    def _extract_color(self, text: str) -> str:
        t = (text or "").lower()
        m = {
            "rojo": "red",
            "azul": "blue",
            "verde": "green",
            "amarillo": "yellow",
            "cian": "cyan",
            "magenta": "magenta",
            "naranja": "orange",
            "morado": "purple",
            "violeta": "violet",
            "rosado": "pink",
            "rosa": "pink",
            "blanco": "white",
            "negro": "black",
            "red": "red",
            "blue": "blue",
            "green": "green",
            "yellow": "yellow",
            "cyan": "cyan",
            "magenta": "magenta",
            "orange": "orange",
            "purple": "purple",
            "pink": "pink",
            "white": "white",
            "black": "black",
        }
        for k, v in m.items():
            if re.search(rf'\b{k}\b', t):
                return v
        mhex = re.search(r'#([0-9a-fA-F]{6})', t)
        if mhex:
            return "#" + mhex.group(1).lower()
        return ""

    def _summarize_context(self, ctx_text: str) -> str:
        s = (ctx_text or "").strip()
        return s[:120] + ("…" if len(s) > 120 else "")

    def _fallback_phrase(self, primary: str, allow_appearance: bool, ctx_snippet: str) -> str:
        base = {
            "feliz": "Te noto con buena energía.",
            "triste": "Noto algo de tristeza; estoy contigo.",
            "enojado": "Percibo tensión; respiramos juntos si quieres.",
            "sorprendido": "¡Qué sorpresa! ¿Hacemos algo breve?",
            "neutral": "Aquí contigo. ¿Te gustaría que haga algo?",
            "no visible": "Estoy contigo. ¿Quieres que haga algo?",
        }
        msg = base.get(primary, "Aquí estoy contigo.")
        if ctx_snippet:
            msg += " " + ctx_snippet
        return msg

    def _emo_color(self, primary: str) -> str:
        return {
            "feliz": "green",
            "triste": "blue",
            "enojado": "red",
            "sorprendido": "yellow",
            "neutral": "white",
            "no visible": "white",
        }.get(primary, "white")

    def _parse_emotions(self, raw: str):
        if not raw:
            return ("neutral", "neutral")
        s = raw.strip().lower()
        m = re.findall(r'(feliz|triste|enojado|sorprendido|neutral|no visible)', s)
        if m:
            return (m[0], m[1] if len(m) > 1 else "neutral")
        return ("neutral", "neutral")

    def _update_mood(self, p: str, s: str):
        delta = {
            "feliz": (+0.2, +0.1),
            "triste": (-0.25, -0.05),
            "enojado": (-0.3, +0.3),
            "sorprendido": (+0.05, +0.25),
            "neutral": (+0.02, +0.0),
            "no visible": (+0.0, +0.0),
        }
        dv1, da1 = delta.get(p, (0, 0))
        dv2, da2 = delta.get(s, (0, 0))
        dv = 0.7 * dv1 + 0.3 * dv2
        da = 0.7 * da1 + 0.3 * da2
        self.mood_valence = max(-1, min(1, 0.85 * self.mood_valence + 0.15 * (self.mood_valence + dv)))
        self.mood_arousal = max(0, min(1, 0.85 * self.mood_arousal + 0.15 * (self.mood_arousal + da)))

    def _describe_mood(self, v: float, a: float) -> str:
        tone = "positivo" if v > 0.2 else ("neutro" if v > -0.2 else "sensible")
        energy = "enérgico" if a > 0.6 else ("tranquilo" if a < 0.35 else "atento")
        return f"ánimo {tone}, {energy}"

    def _intent_fix_motion(self, a: dict, text: str) -> dict:
        t = (text or "").lower()

        def setm(x=None, y=None, th=None):
            if x is not None:
                a["x"] = float(x)
            if y is not None:
                a["y"] = float(y)
            if th is not None:
                a["theta"] = float(th)

        if "gira" in t or "girar" in t:
            if "derecha" in t:
                setm(0, 0, -0.5)
                return a
            if "izquierda" in t:
                setm(0, 0, +0.5)
                return a
        if any(k in t for k in ["adelante", "frente", "avanza", "avanzar"]):
            setm(+0.2, 0, 0)
            return a
        if any(k in t for k in ["atrás", "atras", "retrocede", "retroceder"]):
            setm(-0.2, 0, 0)
            return a
        if "izquierda" in t:
            setm(0, +0.15, 0)
            return a
        if "derecha" in t:
            setm(0, -0.15, 0)
            return a
        if any(k in t for k in ["camina", "caminar", "moverte", "muévete", "muevete", "andar", "ve"]):
            x = float(a.get("x", 0.0))
            y = float(a.get("y", 0.0))
            th = float(a.get("theta", 0.0))
            if abs(y) > abs(x) and abs(x) < 0.05:
                setm(0, 0, th)
            else:
                setm(max(0.15, min(0.25, x if x != 0 else 0.2)), 0, 0)
        return a

    # Normalizador de animaciones
    def normalize_animation_name(self, a: dict):
        if (a.get("type", "") or "").lower() != "play_animation":
            return a
        raw = (a.get("animation_name", "") or "").strip()
        if not raw:
            return None
        low = self._norm(raw)

        # Alias exacto
        if low in {self._norm(k) for k in self.ANIM_ALIASES.keys()}:
            for k, v in self.ANIM_ALIASES.items():
                if self._norm(k) == low:
                    a["animation_name"] = v
                    return a
        # Nombre completo
        if raw in self.ANIM_CATALOG:
            return a
        # Basename
        if low in self.ANIM_BASENAME:
            a["animation_name"] = self.ANIM_BASENAME[low]
            return a
        # Palabras clave
        keywords = {
            "highway": "Sit/Waiting/Music_HighwayToHell_1",
            "highwaytohell": "Sit/Waiting/Music_HighwayToHell_1",
            "acdc": "Sit/Waiting/Music_HighwayToHell_1",
            "happybirthday": "Stand/Waiting/HappyBirthday_1",
            "cumple": "Stand/Waiting/HappyBirthday_1",
        }
        for key, full in keywords.items():
            if key in low:
                a["animation_name"] = full
                return a
        # Coincidencia parcial de basename
        token = low.split('/')[-1]
        matches = [full for base, full in self.ANIM_BASENAME.items() if token in base]
        if len(matches) == 1:
            a["animation_name"] = matches[0]
            return a

        self.get_logger().warning(f"[LLM Bridge] Animación desconocida: '{raw}' → omitida.")
        return None


def main():
    rclpy.init()
    node = LlmBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()