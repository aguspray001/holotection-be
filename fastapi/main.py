from random import random
from fastapi import FastAPI, WebSocket
import uvicorn
from helper import detectFunction

# create apps:
app = FastAPI(title="WebSocket Application")

@app.get("/")
async def root():
    return {"message": "wellcome to hololens websocket"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        respFromClient = await websocket.receive_text()
        print("respFromClient ==>" + respFromClient)
        videoUrl = 'D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/video/video.mp4'
        detect_result = detectFunction.detectWithYoloV3(videoUrl)
        # print("detect_result ==> " + detect_result)
        await websocket.send_json(detect_result, "text")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

# respFromClient = await websocket.receive_text()
# print("respFromClient ==> " + respFromClient)
# wsEvent = json.loads(respFromClient)["eventMsg"]
# videoUrl = 'D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/video/video.mp4'
# detect_result = detectFunction.detectWithYoloV3(videoUrl)
# await websocket.send_json(detect_result, "text")

# if(wsEvent == "img event"):
#     img = useConvertByteToImage(respFromClient);
#     print("img ==> " + img)