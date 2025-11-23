#!/usr/bin/env python3
# -- coding: utf-8 --

import os, json, time, base64, cv2, requests, re
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from naoqi_bridge_msgs.msg import HeadTouch
from std_msgs.msg import String
from cv_bridge import CvBridge

from naoqi_utilities_msgs.msg import LedParameters

ALLOWED = ["feliz", "triste", "enojado", "sorprendido", "neutral", "no visible"]

class EmotionClassifierNode(Node):
    """
    Tap 1: captura frame1 → publica estado RECORDING y ojos AMARILLOS
    Tap 2: captura frame2, clasifica (frame1, frame2), publica emociones y estado REC_DONE con ojos VERDES
    El BehaviorNode señalará INTERACTION_DONE (morado) al terminar el plan.
    """
    def __init__(self):
        super().__init__('emotion_classifier_node')

        # Parámetros
        self.declare_parameter('camera_topic', '/camera/front/image_raw')
        self.declare_parameter('openai_model', 'gpt-4o')
        self.declare_parameter('timeout_sec', 25.0)
        self.declare_parameter('openai_api_key', os.getenv('OPENAI_API_KEY', ''))

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.model        = self.get_parameter('openai_model').get_parameter_value().string_value
        self.timeout_sec  = float(self.get_parameter('timeout_sec').value)
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value


        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.frame1 = None
        self.frame2 = None
        self.tap_count = 0 

        # ROS IO
        self.create_subscription(Image, self.camera_topic, self._on_image, 10)
        self.create_subscription(HeadTouch, '/head_touch', self._on_touch, 10)
        self.pub_emotion = self.create_publisher(String, '/emotion', 10)
        self.pub_state   = self.create_publisher(String, '/interaction/state', 10)
        self.pub_leds    = self.create_publisher(LedParameters, '/set_leds', 10)

        self.get_logger().info(f"[emotion] listo | cam={self.camera_topic} | model={self.model}")

    # Callbacks
    def _on_image(self, msg: Image):
        try:
            self.last_frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"[emotion] error convirtiendo imagen: {e}")

    def _on_touch(self, msg: HeadTouch):
        if msg.state != 1:
            return
        if self.last_frame_bgr is None:
            self.get_logger().warning("[emotion] no hay frame aún.")
            return

        self.tap_count = 1 if self.tap_count >= 2 else self.tap_count + 1

        if self.tap_count == 1:
            # TAP 1: iniciar captura
            self.frame1 = self.last_frame_bgr.copy()
            self.frame2 = None
            self._publish_state("RECORDING")
            self._set_eyes_rgb(255, 255, 0, duration=0.8)  # Acá se activa el color amarillo
            self.get_logger().info("[emotion] TAP1 capturado (esperando TAP2 para clasificar)")
        elif self.tap_count == 2:
            # TAP 2: clasificar
            self._publish_state("REC_DONE")
            self._set_eyes_rgb(0, 255, 0, duration=0.8)  # Acá se activa el color verde
            self.frame2 = self.last_frame_bgr.copy()
            self.get_logger().info("[emotion] TAP2 capturado → clasificando…")

            e1 = self._classify(self.frame1) if self.frame1 is not None else "no visible"
            e2 = self._classify(self.frame2) if self.frame2 is not None else "no visible"
            out = f"[{self._cap(e1)}, {self._cap(e2)}]"
            self.pub_emotion.publish(String(data=out))
            self.get_logger().info(f"[emotion] publicado: {out}")

            # reset para el siguiente par de taps
            
            self.frame1 = None
            self.frame2 = None
            self.tap_count = 0

    def _classify(self, frame_bgr):
        if frame_bgr is None:
            return "no visible"
        
        if not (self.api_key or "").strip():
            return "neutral"
        try:
            data_url = self._bgr_to_data_url(frame_bgr)
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            system_msg = ("Detecta una sola emoción facial en español, "
                          "entre {feliz, triste, enojado, sorprendido, neutral, no visible}. "
                          "Responde SOLO la palabra.")
            user_content = [
                {"type":"text","text":"Emoción principal:"},
                {"type":"image_url","image_url":{"url":data_url}},
            ]
            payload = {
                "model": self.model, "temperature": 0.0,
                "messages": [{"role":"system","content":system_msg},{"role":"user","content":user_content}],
                "max_tokens": 6
            }
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            r.raise_for_status()
            s = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
            return self._normalize(s)
        except Exception as e:
            self.get_logger().warning(f"[emotion] fallo LLM: {e}")
            return "neutral"

    def _bgr_to_data_url(self, frame_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok: raise RuntimeError("jpeg fail")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _normalize(self, s):
        s = (s or "").strip().lower()
        m = {
            "feliz": ["feliz","contento","alegre","happy"],
            "triste": ["triste","sad"],
            "enojado": ["enojado","enfadado","molesto","angry"],
            "sorprendido": ["sorprendido","sorpresa","surprised"],
            "neutral": ["neutral"],
            "no visible": ["no visible","sin rostro","no face","none","unknown"],
        }
        for k, arr in m.items():
            if s in arr: return k
        for k in ALLOWED:
            if k in s: return k
        return "neutral"

    def _cap(self, s): 
        return s.capitalize() if s else "Desconocida"

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


def main():
    rclpy.init()
    node = EmotionClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()