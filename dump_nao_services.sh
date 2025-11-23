#!/bin/bash

OUTPUT="nao_services.txt"

# Lista de namespaces que nos interesan
NAMESPACES=("naoqi_speech_node" "naoqi_navigation_node" "naoqi_miscellaneous_node" "naoqi_perception_node")

echo "📋 Servicios NAO (Speech, Navigation, Miscellaneous, Perception)" > $OUTPUT
echo "==============================================================" >> $OUTPUT

for srv in $(ros2 service list); do
    for ns in "${NAMESPACES[@]}"; do
        if [[ "$srv" == "$ns" ]]; then
            echo -e "\n🔹 Servicio: $srv" >> $OUTPUT
            srv_type=$(ros2 service type $srv 2>/dev/null)
            if [ -n "$srv_type" ]; then
                echo "   Tipo: $srv_type" >> $OUTPUT
                echo "   Definición:" >> $OUTPUT
                ros2 interface show $srv_type >> $OUTPUT 2>/dev/null
            else
                echo "   ❌ No se pudo obtener tipo" >> $OUTPUT
            fi
        fi
    done
done

echo -e "\n✅ Archivo generado en $OUTPUT"
