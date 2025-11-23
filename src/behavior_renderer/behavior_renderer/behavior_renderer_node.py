#!/usr/bin/env python3
# -- coding: utf-8 --
import json, queue, threading, time, math, re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Point

from naoqi_utilities_msgs.msg import LedParameters
from naoqi_utilities_msgs.srv import Say, MoveTo, PointAt, GoToPosture, PlayAnimation


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # Suscripción al plan
        self.create_subscription(String, '/llm/plan', self._on_plan, 10)

        self.create_subscription(String, '/interaction/state', self._on_state, 10)

        self.pub_state = self.create_publisher(String, '/interaction/state', 10)

        # Clientes de servicios NAOqi
        self.cli_say            = self.create_client(Say, '/naoqi_speech_node/say')
        self.cli_move_to        = self.create_client(MoveTo, '/naoqi_navigation_node/move_to')
        self.cli_point_at       = self.create_client(PointAt, '/naoqi_perception_node/point_at')
        self.cli_go_to_posture  = self.create_client(GoToPosture, '/naoqi_manipulation_node/go_to_posture')
        self.cli_play_animation = self.create_client(PlayAnimation, '/naoqi_manipulation_node/play_animation')

        self.pub_leds = self.create_publisher(LedParameters, '/set_leds', 10)
        self.cli_toggle_awareness = self.create_client(SetBool, '/naoqi_miscellaneous_node/toggle_awareness')

        # Cola de acciones
        self.action_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_actions, daemon=True)
        self.worker_thread.start()

        self.current_plan_id = None
        self.current_posture = None  

        self.get_logger().info("✅ BehaviorNode listo (voz/caminar/ojos/posturas/animaciones)")

    # Plan
    def _on_plan(self, msg: String):
        try:
            plan = json.loads(msg.data.strip())
        except Exception as e:
            self.get_logger().error(f"❌ Plan no es JSON válido: {e}")
            return

        pid = plan.get("meta", {}).get("plan_id")
        if pid is not None and pid != self.current_plan_id:
            # Limpia cola anterior
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                    self.action_queue.task_done()
                except Exception:
                    break
            self.current_plan_id = pid
            self.get_logger().info(f"📦 Nuevo plan_id={pid} (cola limpiada)")

        actions = plan.get("actions")
        if not isinstance(actions, list):
            if isinstance(plan, dict) and "say" in plan and isinstance(plan["say"], str):
                actions = [{"type": "say", "text": plan["say"]}]
            else:
                self.get_logger().warning("⚠ Plan sin 'actions'; ignorado.")
                return

        actions = self._preprocess_actions(actions)

        for act in actions:
            self.action_queue.put(act)

        self.get_logger().info(f"📥 Encoladas {self.action_queue.qsize()} acción(es) (plan_id={self.current_plan_id})")

    def _on_state(self, msg: String):
        pass

    # Procesar aciones
    def _preprocess_actions(self, actions):
        """
        - Elimina duplicados exactos consecutivos.
        - Inserta go_to_posture requerida antes de play_animation/move_to si hace falta.
        - Colapsa cadenas de go_to_posture dejando solo la necesaria.
        - *Respeta explicit=True* (no se colapsa ni se descarta).
        """
        raw = []
        for a in actions:
            if isinstance(a, dict) and "type" in a:
                raw.append(a)
            else:
                self.get_logger().warning(f"⚠ Acción inválida: {a}")

        dedup = []
        prev = None
        for a in raw:
            if a == prev:
                continue
            dedup.append(a)
            prev = a

        expanded = []
        posture_shadow = self.current_posture 

        def need_posture_for(act):
            t = (act.get("type") or "").lower()
            if t == "move_to":
                return "Stand"
            if t == "play_animation":
                an = str(act.get("animation_name", "") or "")
                return self._infer_posture_from_anim(an)
            return None

        for a in dedup:
            t = (a.get("type") or "").lower()

            if t == "go_to_posture":
                name = (a.get("name") or "").strip()
                if not name:
                    continue
                if bool(a.get("explicit", False)):
                    expanded.append({"type": "go_to_posture", "name": name, "explicit": True})
                    posture_shadow = name
                    continue
                if posture_shadow and name == posture_shadow:
                    continue
                expanded.append({"type": "go_to_posture", "name": name})
                posture_shadow = name
                continue

            req = need_posture_for(a)
            if req and req != posture_shadow:
                expanded.append({"type": "go_to_posture", "name": req})
                posture_shadow = req

            expanded.append(a)

        compact = []
        last_gtp = None
        last_gtp_explicit = False

        for a in expanded:
            t = (a.get("type") or "").lower()
            if t == "go_to_posture" and not a.get("explicit", False):
                last_gtp = a
                last_gtp_explicit = False
                continue
            if t == "go_to_posture" and a.get("explicit", False):
                if not compact or compact[-1] != a:
                    compact.append(a)
                last_gtp = None
                last_gtp_explicit = False
                continue

            if last_gtp is not None and not last_gtp_explicit:
                req = need_posture_for(a)
                if req is None or req == last_gtp.get("name"):
                    if not compact or compact[-1] != last_gtp:
                        compact.append(last_gtp)
                last_gtp = None
                last_gtp_explicit = False

            compact.append(a)

        if last_gtp is not None and not last_gtp_explicit:
            if self.current_posture != last_gtp.get("name"):
                compact.append(last_gtp)

        final = []
        prev = None
        for a in compact:
            if a == prev:
                continue
            final.append(a)
            prev = a

        return final

    # Procesamiento de acciones
    def _process_actions(self):
        while rclpy.ok():
            act = self.action_queue.get()
            try:
                self._execute_action(act)
            except Exception as e:
                self.get_logger().error(f"❌ Error ejecutando acción {act}: {e}")
            finally:
                self.action_queue.task_done()

            if self.action_queue.empty() and self.current_plan_id is not None:
                # Ojos morados de finalización
                self._publish_state("INTERACTION_DONE")
                self._set_eyes_rgb(128, 0, 128, duration=0.8) 
                # Plan procesado
                self.current_plan_id = None

    # Ejecutores
    def _ensure_posture(self, needed: str, *, force: bool = False):
        if not needed:
            return
        if not force and self.current_posture == needed:
            return
        req = GoToPosture.Request()
        req.posture_name = needed
        self._call_srv(self.cli_go_to_posture, req, f"GO_TO_POSTURE '{needed}' (force={force})")
        self.current_posture = needed
        time.sleep(0.8)

    def _execute_action(self, action: dict):
        at = (action.get("type") or "").lower()

        # SAY
        if at == "say":
            text = str(action.get("text", ""))
            req = Say.Request()
            req.text = text
            lang_from_plan = str(action.get("language", "") or "").lower()
            if lang_from_plan in ("es", "es-es", "es_es", "español", "spanish", "spa"):
                req.language = "Spanish"
            else:
                req.language = ""  # por defecto en NAO
            req.animated = bool(action.get("animated", False))
            req.asynchronous = bool(action.get("async", True))
            self._call_srv(self.cli_say, req, f"SAY '{self._short(text)}'")
            time.sleep(min(3.0, max(1.0, len(text) / 18.0)))
            return

        # SET_LEDS
        if at in ("set_leds", "leds", "eyes", "set_eyes_color"):
            name = str(action.get("name", "") or "FaceLeds")
            r, g, b = self._parse_color(action.get("color"))
            for k in ("red", "green", "blue"):
                if k in action:
                    if k == "red":   r = int(action["red"])
                    if k == "green": g = int(action["green"])
                    if k == "blue":  b = int(action["blue"])
            try:
                duration = float(action.get("duration", 0.5))
            except Exception:
                duration = 0.5
            duration = max(0.05, min(2.0, duration))
            self._set_eyes_rgb(r, g, b, duration, name=name)
            return

        # POINT_AT
        if at == "point_at":
            self._ensure_posture("Stand")
            effector = str(action.get("effector", "RArm"))
            try:
                x = float(action.get("x", 0.3))
                y = float(action.get("y", 0.0))
                z = float(action.get("z", 0.9))
            except Exception:
                self.get_logger().warning(f"⚠ Parámetros inválidos en point_at: {action}")
                return
            frame_str = str(action.get("frame", "TORSO")).upper()
            frame_map = {"TORSO": 0, "WORLD": 1, "ROBOT": 2}
            frame = frame_map.get(frame_str, 0)
            try:
                speed = float(action.get("speed", 0.3))
            except Exception:
                speed = 0.3
            speed = max(0.05, min(0.8, speed))
            if self.cli_toggle_awareness.wait_for_service(timeout_sec=0.2):
                try:
                    off = SetBool.Request()
                    off.data = False
                    self._call_srv(self.cli_toggle_awareness, off, "TOGGLE_AWARENESS -> False")
                except Exception:
                    pass
            req = PointAt.Request()
            req.effector_name = effector
            req.point = Point(x=x, y=y, z=z)
            req.frame = frame
            req.speed = speed
            self._call_srv(self.cli_point_at, req, f"POINT_AT {effector} -> ({x:.2f},{y:.2f},{z:.2f}) [{frame_str}] v={speed:.2f}")
            if self.cli_toggle_awareness.wait_for_service(timeout_sec=0.2):
                try:
                    on = SetBool.Request()
                    on.data = True
                    self._call_srv(self.cli_toggle_awareness, on, "TOGGLE_AWARENESS -> True")
                except Exception:
                    pass
            time.sleep(1.0)
            return

        # GO_TO_POSTURE
        if at == "go_to_posture":
            name_raw = str(action.get("name", "")).strip()
            if not name_raw:
                self.get_logger().warning("⚠ go_to_posture sin 'name'; ignoro")
                return
            force_flag = bool(action.get("explicit", False))
            self._ensure_posture(name_raw, force=force_flag)
            return

        # MOVE_TO
        if at == "move_to":
            self._ensure_posture("Stand")
            try:
                x = float(action.get("x", 0.0))
                y = float(action.get("y", 0.0))
                theta = float(action.get("theta", 0.0))
            except Exception:
                self.get_logger().warning(f"⚠ Parámetros inválidos para move_to: {action}")
                return
            req = MoveTo.Request()
            req.x_coordinate, req.y_coordinate, req.theta_coordinate = x, y, theta
            self._call_srv(self.cli_move_to, req, f"MOVE_TO (x={x:.2f}, y={y:.2f}, th={theta:.2f})")
            dist = math.hypot(x, y)
            est = dist / 0.18 + abs(theta) / 0.45
            time.sleep(max(1.2, est))
            return

        # PLAY_ANIMATION
        if at == "play_animation":
            an = str(action.get("animation_name", "")).strip()
            if not an:
                self.get_logger().warning("⚠ play_animation sin 'animation_name'; ignoro")
                return

            need = self._infer_posture_from_anim(an) or "Rest"
            self._ensure_posture(need, force=True)

            req = PlayAnimation.Request()
            req.animation_name = an
            self._call_srv(self.cli_play_animation, req, f"PLAY_ANIMATION '{an}'")
            time.sleep(1.0)
            return

        self.get_logger().warning(f"⚠ Acción desconocida o no soportada: {at}")

    # Utilities
    def _infer_posture_from_anim(self, anim_name: str):
        if not anim_name:
            return None
        if anim_name.startswith("Stand/"):
            return "Stand"
        if anim_name.startswith("Sit/"):
            return "Sit"
        if anim_name.startswith("Rest/"):
            return "Rest"
        m = re.search(r'(^|/)(stand|sit|rest)(/|$)', anim_name, re.IGNORECASE)
        if not m:
            return None
        token = m.group(2).lower()
        return "Stand" if token == "stand" else ("Sit" if token == "sit" else "Rest")

    def _call_srv(self, client, request, desc: str):
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"⚠ Servicio no disponible para {desc}")
            return
        self.get_logger().info(f"▶ {desc}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None:
            self.get_logger().error(f"❌ Falló {desc} (sin respuesta)")
            return
        succ = getattr(resp, "success", None)
        msg  = getattr(resp, "message", None)
        if succ is not None:
            if succ:
                self.get_logger().info(f"✅ {desc} OK")
            else:
                self.get_logger().warning(f"⚠ {desc} success={succ} msg='{msg}'")
        else:
            self.get_logger().info(f"✅ {desc} completado")

    def _publish_state(self, st: str):
        self.pub_state.publish(String(data=st))

    def _set_eyes_rgb(self, r, g, b, duration=0.5, name="FaceLeds"):
        msg = LedParameters()
        msg.name, msg.red, msg.green, msg.blue, msg.duration = (
            name,
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
            float(max(0.05, min(2.0, duration))),
        )
        self.get_logger().info(f"▶ SET_LEDS {name} RGB=({msg.red},{msg.green},{msg.blue}) dur={msg.duration:.2f}s")
        self.pub_leds.publish(msg)

    def _parse_color(self, val):
        if val is None:
            return (255, 255, 255)
        if isinstance(val, str):
            s = val.strip().lower()
            if s.startswith("#") and len(s) == 7:
                try:
                    return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
                except Exception:
                    return (255, 255, 255)
            names = {
                "white": (255, 255, 255), "black": (0, 0, 0),
                "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
                "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
                "violet": (148, 0, 211), "brown": (165, 42, 42),
                "blanco": (255, 255, 255), "negro": (0, 0, 0),
                "rojo": (255, 0, 0), "verde": (0, 255, 0), "azul": (0, 0, 255),
                "amarillo": (255, 255, 0), "cian": (0, 255, 255), "magenta": (255, 0, 255),
                "naranja": (255, 165, 0), "morado": (128, 0, 128), "rosado": (255, 192, 203),
                "violeta": (148, 0, 211), "marron": (165, 42, 42), "marrón": (165, 42, 42)
            }
            return names.get(s, (255, 255, 255))
        if isinstance(val, (list, tuple)) and len(val) == 3:
            try:
                r, g, b = float(val[0]), float(val[1]), float(val[2])
                if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
                    r, g, b = r * 255.0, g * 255.0, b * 255.0
                return (int(r), int(g), int(b))
            except Exception:
                return (255, 255, 255)
        if isinstance(val, dict):
            try:
                r = float(val.get("r", val.get("red", 255)))
                g = float(val.get("g", val.get("green", 255)))
                b = float(val.get("b", val.get("blue", 255)))
                if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
                    r, g, b = r * 255.0, g * 255.0, b * 255.0
                return (int(r), int(g), int(b))
            except Exception:
                return (255, 255, 255)
        return (255, 255, 255)

    def _short(self, s: str, n: int = 60) -> str:
        return s if len(s) <= n else s[:n-1] + "…"


def main():
    rclpy.init()
    node = BehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()