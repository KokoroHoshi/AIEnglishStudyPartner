from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    AudioMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    ButtonsTemplate,
    MessageAction,
    TemplateMessage
)


import time
import requests

import os
from dotenv import load_dotenv

from llm import LLM
from vlm import VLM
from stt import STT
from tts import TTS
from rag import TextEmbeddingModel, RAG

load_dotenv()

NGROK_TOKEN = os.getenv('NGROK_TOKEN')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
HF_TOKEN = os.getenv('HF_TOKEN')

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# llm_id = "MaziyarPanahi/Llama-3-8B-Instruct-v0.8"
# cache_dir = "../llm/model"

cache_dir = "./models"
llm_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
vlm_id = "google/paligemma-3b-mix-224"
stt_id = "openai/whisper-large-v3"
embedding_id = 'intfloat/multilingual-e5-large-instruct'

llm = LLM(llm_id=llm_id, cache_dir=cache_dir, hf_token=HF_TOKEN)
vlm = VLM(model_id=vlm_id, cache_dir=cache_dir, hf_token=HF_TOKEN)
stt = STT(model_id=stt_id, cache_dir=cache_dir, hf_token=HF_TOKEN)
tts = TTS()
embedding_model = TextEmbeddingModel(embedding_id, cache_dir)
rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup event
    global llm, stt, tts
    # global vlm
    global embedding_model, rag

    auto_update_webhook_url(8080)

    llm.load()
    # vlm.load()
    stt.load()
    tts.load()

    embedding_model.load()
    index_dir = './vector_db/indices'
    text_dir = './vector_db/texts'
    rag = RAG(embedding_model, index_dir, text_dir)
    yield
    
    # shutdown event
    llm.db.close()

app = FastAPI(lifespan=lifespan)

def print_gpu_memory():
    from torch.cuda import memory_allocated, memory_reserved

    allocated = memory_allocated()
    reserved = memory_reserved()
    print(f"Allocated: {allocated / (1024 ** 3):.2f} GB")
    print(f"Reserved: {reserved / (1024 ** 3):.2f} GB")

# tmp /because cannot find how to get image with linebot v3 (there is no get_message_content in v3) 
def get_message_content(message_id: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    return response.content

def auto_update_webhook_url(port: int):
    from pyngrok import ngrok

    ngrok.set_auth_token(NGROK_TOKEN)
    ngrok_url = ngrok.connect(port).public_url
    print(f"Ngrok URL: {ngrok_url}")

    line_put_endpoint_url = "https://api.line.me/v2/bot/channel/webhook/endpoint"
    data = {"endpoint": ngrok_url + '/callback'}
    headers = {
        "Authorization": "Bearer " + LINE_CHANNEL_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    while True:
        time.sleep(1)
        res = requests.put(line_put_endpoint_url, headers=headers, json=data)
        if res.status_code == 200:
            print("Webhook URL updated successfully")
            break
        else:
            print(f"Error. Status code: {res.status_code}")

@app.post("/callback")
async def callback(request: Request):
    # 獲取 X-Line-Signature 標頭
    signature = request.headers["X-Line-Signature"]

    # 獲取請求內容
    body = await request.body()

    try:
        # 驗證簽名
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return JSONResponse(content={"status": "OK"})

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    global llm    

    # 使用 ApiClient 來發送回覆
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        user_id = event.source.user_id
        user_profile = line_bot_api.get_profile(user_id)
        user_name = user_profile.display_name
        user_profile_photo = user_profile.picture_url
        user_status_message = user_profile.status_message

        if not llm.db.exists("user", user_id):
            llm.db.insert_data('user', 'user_id, name, profile_photo, status_message',
                            (user_id, user_name, user_profile_photo, user_status_message))
            llm.db.insert_data('parameter', 'user_id, english_level, current_mode, conversation_history', 
                               (user_id, llm.levels[0], llm.modes[0], '[]'))

        
        user_mode = llm.get_user_mode(user_id)
        user_msg = event.message.text
        reply_msgs=[]
        reply_text = ""

        if llm.mode_check(user_msg):
            reply_text = llm.change_mode(user_id, user_msg)

            # 程度設置
            if user_mode == llm.modes[5]:
                reply_msgs.append(TextMessage(text=f"目前設置的程度: {llm.get_user_level(user_id)[1:]}"))
                btn_template = ButtonsTemplate(
                    title='程度設置',
                    text='請選擇想要設置的程度',
                    actions=[
                        MessageAction(label='A1-A2', text='$A1-A2'),
                        MessageAction(label='B1-B2', text='$B1-B2'),
                        MessageAction(label='C1-C2', text='$C1-C2'),
                        MessageAction(label='不清楚自己的程度', text='不清楚自己的程度'),
                    ]
                )
                reply_msgs.append(TemplateMessage(alt_text='程度設置', template=btn_template))

        else:
            rag_result = None
            do_infer = True

            # 學習資源
            if user_mode == llm.modes[4]:
                rag_result = rag.retrieve_by_id(id='learning_resource', query=user_msg)
                # print(rag_result)
            
            # 程度設置
            if user_mode == llm.modes[5]:
                if llm.level_check(user_msg):
                    reply_text = llm.change_level(user_id, user_msg)
                    do_infer = False
            
            if do_infer:
                reply_text = llm.infer_with_db(user_id, user_msg, rag_infomation=rag_result)

        if not reply_text:
            reply_text = "抱歉目前這個LINE機器人有點問題。 Sorry, there are some problems with this line bot."
        
        # reply_msgs.append(TextMessage(text=reply_text))
        reply_msgs.insert(0, TextMessage(text=reply_text))
        
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=reply_msgs
            )
        )
        # print("message sended")

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    global llm, vlm

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        user_id = event.source.user_id
        user_profile = line_bot_api.get_profile(user_id)
        user_name = user_profile.display_name
        user_profile_photo = user_profile.picture_url
        user_status_message = user_profile.status_message

        if not llm.db.exists("user", user_id):
            llm.db.insert_data('user', 'user_id, name, profile_photo, status_message',
                            (user_id, user_name, user_profile_photo, user_status_message))
            llm.db.insert_data('parameter', 'user_id, english_level, current_mode, conversation_history', 
                               (user_id, llm.levels[0], llm.modes[0], '[]'))
        
        image_bytes = get_message_content(event.message.id)

        # save image
        # jpg_file = f"./tmp/user_image.jpg"
        # with open(jpg_file, 'wb') as fd:
        #     fd.write(image_content)
        # print(f"save image to {jpg_file}")
        
        image_description = ""
        image_description = vlm.infer(image_bytes=image_bytes, prompt="What is shown in this image?")
        print(image_description)

        reply_text = ""
        if not image_description:
            reply_text = "抱歉目前這個LINE機器人有點問題。 Sorry, there are some problems with this line bot."
        else:
            reply_text = llm.infer_with_memory(user_id, f"這裡有張圖片，圖片描述如下{image_description}")

            if not reply_text:
                reply_text = "抱歉目前這個LINE機器人有點問題。 Sorry, there are some problems with this line bot."


        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )
        # print("message sended")

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio_message(event: MessageEvent):
    global llm, stt, tts

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        user_id = event.source.user_id
        user_profile = line_bot_api.get_profile(user_id)
        user_name = user_profile.display_name
        user_profile_photo = user_profile.picture_url
        user_status_message = user_profile.status_message

        if not llm.db.exists("user", user_id):
            llm.db.insert_data('user', 'user_id, name, profile_photo, status_message',
                            (user_id, user_name, user_profile_photo, user_status_message))
            llm.db.insert_data('parameter', 'user_id, english_level, current_mode, conversation_history', 
                               (user_id, llm.levels[0], llm.modes[0], '[]'))
        
        audio_bytes = get_message_content(event.message.id)
    
        # save to wav
        output_file_path = "./tmp/user_audio.wav"
        stt.save_m4a_bytes_to_wav(audio_bytes, output_file_path)
        
        stt_result = ""
        stt_result = stt.infer(output_file_path)
        print(stt_result['text'])
        tts.infer(stt_result['text'])

        reply_text = ""
        if not stt_result:
            reply_text = "抱歉目前這個LINE機器人有點問題。 Sorry, there are some problems with this line bot."
        else:
            reply_text = llm.infer_with_memory(user_id, f"以下是學生的語音訊息{stt_result}", prompt_role="學生音檔")

            if not reply_text:
                reply_text = "抱歉目前這個LINE機器人有點問題。 Sorry, there are some problems with this line bot."


        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )
        # print("message sended")


if __name__ == "__main__":
    # testing
    # llm.load()

    # embedding_model.load()
    # index_dir = './vector_db/indices'
    # text_dir = './vector_db/texts'
    # rag = RAG(embedding_model, index_dir, text_dir)

    # user_id = 'test_3'
    
    # i = 0

    # user_input = input("Student: ")
    # while user_input != "q":
    #     # if user_input == "h":
    #     #     llm.show_abstraction()

    #     rag_result = rag.retrieve_by_user_id(user_id, user_input)

    #     if rag_result is None:
    #         result = llm.infer_with_memory(user_input)
    #     else:
    #         result = llm.infer_with_rag_zh(rag_result, user_input)
    #     print(f"AI English teacher: {result}")
        
    #     user_input = input("Student: ")

    # # print()
    # # print(f"history: {llm.conversation_history}")
    # # print()

    # rag.save_conversations_by_user_id(user_id, llm.conversation_history)

    # abstraction = llm.abstract(llm.conversation_history)


    # running
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)