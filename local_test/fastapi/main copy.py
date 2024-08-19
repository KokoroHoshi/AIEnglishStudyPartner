from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)

from pyngrok import ngrok
import requests
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, BitsAndBytesConfig

app = FastAPI()

NGROK_TOKEN = '2SDRFc7QfhCgf0BXFpFYcZ3WGBi_62JVvkLaG5hVsW4QouEDU'

# 替換為你的 Channel Access Token 和 Channel Secret
LINE_CHANNEL_ACCESS_TOKEN = 'Gmus08WrSXXFQQNI/sv7/2dmZTXo0Mjqi0w42CxcBvh/Y9NeQzQxIgzAH4Eeud5yc6u9GVh6bhBdutqGdEFdnyo6Vk7Doj5asdmbkRk2eDiiM7pCUPnriw9iVJARnq9W7GxylYhkLSJnS68x57eFHQdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '81c8ba8c336d45e56fd1783c32a3b441'

# 設定配置
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


llm_id = "MaziyarPanahi/Llama-3-8B-Instruct-v0.8"
cache_dir = "../llm/model"

llm_tokenizer = None
llm_pipeline = None

def load_llm():
    global llm_tokenizer, llm_pipeline
    
    llm_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        llm_id,
        quantization_config=llm_quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=llm,
        tokenizer=llm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

async def llm_inference(prompt, system_prompt=
        """
        You are an AI English teacher teaching a student who is a Chinese speaker.
        If your student ask, you should speak Chinese to him or her to teach his or her English.
        """
):
    global llm_tokenizer, llm_pipeline

    if llm_tokenizer is None or llm_pipeline is None:
        print("LLM is not loaded.")
        return 

    llm_messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{prompt}"},
    ]

    llm_prompt = llm_tokenizer.apply_chat_template(
        llm_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    llm_terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    llm_outputs = llm_pipeline(
        llm_prompt,
        max_new_tokens=512,
        eos_token_id=llm_terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )

    return llm_outputs[0]["generated_text"][len(llm_prompt):]

def auto_update_webhook_url(port):
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
async def handle_text_message(event: MessageEvent):
    reply_text = await llm_inference(event.message.text)
    print(reply_text)

    # 使用 ApiClient 來發送回覆
    async with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        await line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

if __name__ == "__main__":
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    load_llm()

    port = 8080

    auto_update_webhook_url(port)

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
