Human Detection with MQTT Integration






![Human-Detection-by-Facial-Recognition-1](https://github.com/user-attachments/assets/71e19a39-9fe6-4069-a5dc-d3e8afd2eaf7)

![image](https://github.com/user-attachments/assets/8e8d17bf-a1f4-4b61-9a9a-1bdfa2f50664)

Overview

This project detects humans using an IP camera and a YOLO model, then publishes real-time updates to an MQTT broker. It can also turn off appliances if no humans are detected for 20 seconds.
Prerequisites

Before running the script, install the required libraries:

pip install opencv-python numpy paho-mqtt

Dependencies:

    opencv-python → Image processing
    numpy → Array manipulations
    paho-mqtt → MQTT communication
    threading (built-in) → Runs video stream in a separate thread
    time (built-in) → Manages time-based events

How the Code Works
1. Video Capture

    Captures video from an IP camera (mobile camera).
    Runs the stream in a separate thread for better performance.

cap = VideoStream(src=phone_camera_url).start()

2. MQTT Connection

    Connects to the MQTT broker (HiveMQ).
    Sends messages based on human detection.

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.tls_set()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

3. YOLO Model for Human Detection

    Loads the YOLO model weights and configuration.
    Detects humans in the frame and filters out other objects.

net = cv2.dnn.readNet(weights_path, config_path)
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
person_idx = classes.index("person")

4. Human Detection & MQTT Publishing

    If humans are detected, publishes "humans detected".
    If no humans are found for 20 seconds, publishes "humans not detected" and "turn off all appliances".

if human_count > 0:
    mqtt_client.publish(MQTT_TOPIC, "humans detected")
else:
    mqtt_client.publish(MQTT_TOPIC, "humans not detected")
    mqtt_client.publish(MQTT_TOPIC, "turn off all appliances")

5. Displaying Video Output

    Draws bounding boxes around detected humans.
    Displays the number of humans detected.

cv2.putText(resized_frame, f"Humans Detected: {human_count}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Human Detection", resized_frame)

Running the Project

    Install an IP camera app on your phone (e.g., IP Webcam for Android).
    Run the script and enter your phone’s IP address when prompted.
    The program detects humans and sends MQTT messages.

Expected MQTT Messages
Condition	MQTT Message
When humans are detected	"humans detected"
When no humans for 20 seconds	"humans not detected"
To turn off appliances	"turn off all appliances"
Periodic human count update	"Human count: X"
Stopping the Program

    Press q to quit.
    The script will stop the camera stream and disconnect from MQTT before exiting.

except KeyboardInterrupt:
    cap.stop()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    cv2.destroyAllWindows()

License

This project is open-source and available under the MIT License.

This README format is GitHub-friendly, easy to read, and well-structured for your repository. 🚀

