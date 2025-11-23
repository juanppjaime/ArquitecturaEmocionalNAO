#!/usr/bin/env python3
# -- coding: utf-8 --

import time, requests, cv2, base64
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from naoqi_bridge_msgs.msg import HeadTouch
from std_msgs.msg import String
from cv_bridge import CvBridge


class ContextNode(Node):
    """
    En TAP 1 toma un frame y publica un PÁRRAFO en /context/summary.
    Solo texto corrido en español (40–70 palabras).
    """
    def __init__(self):
        super().__init__('context_node')

        self.declare_parameter('camera_topic', '/camera/front/image_raw')
        self.declare_parameter('openai_model', 'gpt-4o')
        self.declare_parameter('timeout_sec', 35.0)
        self.declare_parameter('openai_api_key', os.getenv('OPENAI_API_KEY', ''))

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.model        = self.get_parameter('openai_model').get_parameter_value().string_value
        self.timeout_sec  = float(self.get_parameter('timeout_sec').value)
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value


        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.tap_count = 0  

        self.create_subscription(Image, self.camera_topic, self._on_image, 10)
        self.create_subscription(HeadTouch, '/head_touch', self._on_touch, 10)
        self.pub_context = self.create_publisher(String, '/context/summary', 10)

        self.get_logger().info(f"[context] listo | cam={self.camera_topic} | model={self.model}")

    def _on_image(self, msg: Image):
        try:
            self.last_frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f"[context] error img: {e}")

    def _on_touch(self, msg: HeadTouch):
        if msg.state != 1:
            return
        self.tap_count = 1 if self.tap_count >= 2 else self.tap_count + 1

        if self.tap_count == 1:
            
            if self.last_frame_bgr is None:
                self.get_logger().warning("[context] sin frame para TAP1")
                return
            frame = self.last_frame_bgr.copy()
            self.get_logger().info("[context] TAP1 capturado → describiendo entorno/persona…")
            try:
                paragraph = self._describe(frame)
            except Exception as e:
                self.get_logger().warning(f"[context] LLM falló: {e}")
                paragraph = "Descripción breve: entorno poco claro por iluminación; estoy atento a ti."
                
            preview = paragraph if len(paragraph) <= 240 else paragraph[:237] + "…"
            self.get_logger().info(f"[context] resumen a publicar: {preview}")
            self.pub_context.publish(String(data=paragraph))
            self.get_logger().info("[context] publicado /context/summary")
        elif self.tap_count == 2:
            
            pass

    def _describe(self, frame_bgr) -> str:
        if not (self.api_key or "").strip():
            return "Descripción breve del entorno y de la persona no disponible por ahora."
        data_url = self._bgr_to_data_url(frame_bgr)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        system_msg = (
            "Eres un analista visual responsable. Responde SOLO en español, en UN PÁRRAFO (40–70 palabras). "
            "No infieras identidad/etnia/religión/salud. Describe de forma general: "
            "qué parece haber al fondo (objetos/ambiente), si la persona parece hombre/mujer y adulta/niña, "
            "una estimación gruesa de edad, y un par de detalles de ropa/colores si son evidentes. "
            "Evita disculpas o frases como 'no puedo ver'. Si algo no es claro, dilo brevemente y sigue."
        )
        user_content = [
            {"type":"text","text":"Describe brevemente el entorno y a la persona (un párrafo)."},
            {"type":"image_url","image_url":{"url":data_url}},
        ]
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role":"system","content":system_msg},{"role":"user","content":user_content}],
            "max_tokens": 220
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    def _bgr_to_data_url(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("jpeg fail")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


def main():
    rclpy.init()
    node = ContextNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()