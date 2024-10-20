import asyncio
import os
from dotenv import load_dotenv
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamFace

async def main():
    load_dotenv()
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    client = AsyncHumeClient(api_key=HUME_API_KEY)

    model_config = Config(face=StreamFace())

    stream_options = StreamConnectOptions(config=model_config)

    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        # Simulate a 3-second video stream
        for _ in range(3):
            result = await socket.send_text("Simulated video frame data")
            
            if result.face and result.face.predictions:
                top_emotion = max(result.face.predictions[0]['emotions'], key=lambda x: x['score'])
                print(f"Top emotion: {top_emotion['name']} with score: {top_emotion['score']}")
            
            await asyncio.sleep(1)  # Wait for 1 second before sending the next frame

if __name__ == "__main__":
    asyncio.run(main())