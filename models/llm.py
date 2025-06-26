from pathlib import Path
from typing import Optional
from re import sub, match

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, BitsAndBytesConfig

from models.relationalDB import RelationalDB

import uuid
from datetime import datetime, timezone

class LLM:
    def __init__(self, llm_id: str, cache_dir: str, hf_token: Optional[str] = None, db_instance: RelationalDB = None):
        self.llm_id = llm_id
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.model_loaded = False
        self.max_history_length = 6 # it is better to set an odd 
        
        # Relational Database
        self.db = db_instance

        self.levels = ["$A1-A2", "$B1-B2", "$C1-C2"]
        self.level_descriptions = {
            f"{self.levels[0]}":"""
            A1-A2(零基礎-初學階段)
            這時學生還不太會英文，請多用中文進行教學，並建議學生先注重發音與單字的學習，推薦學生看看網路上英文發音的教學影片，並搜尋英文最常用的一千個單字，用自己的方式背看看

            你的主要目標：
            給學生學習的方向(發音與單字)，幫助學生先背基本的一千個英文單字，以及學習基本的美式英語的發音

            重要注意事項：
            多使用中文教學
            """,
            f"{self.levels[1]}":"""
            B1-B2(中階-中高階段)
            這時學生有一點英文基礎，教學時可以嘗試少用一點中文，依照學生學習喜好，可以建議學生再背更多單字，多聽多看適合自己程度的英文影片，或者找一對一的英文老師練習多使用英文(找你就是個不錯的選擇)，可以用主題式的聊天或閒聊，讓老師再與學生英文交談時，指出學生英文使用上的問題以及如何用得更加流暢

            你的主要目標：
            讓學生摸索自己的學習方式(偏向自學或與人互動)，嘗試增加學生英文交談能力，點出學生使用英文時的問題，以及改善的方式
            """,
            f"{self.levels[2]}":"""
            C1-C2(精通階段)
            這時學生英文基本上很厲害了，基本上都用英文和學生溝通也沒問題。學生可能是再更加貼近母語人士，或是學習特定專業領域的英文，可以先了解學生的學習動機，再考慮教學方向。

            主要目標：
            了解學生的學習動機，以貼近母語人士，或特定專業領域人士的英文來精進學生的英文能力
            """
        }

        self.modes = ["*文法教學", "*寫作建議", "*發音反饋", "*情境對話", "*資源推薦", "*設定頁面"]
        self.system_prompts = {
            f"{self.modes[0]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入文法教學模式，學生可以提出想學習的文法點，或者你可以根據他們的程度推薦一些適合的文法知識。
                你會根據學生的需求提供相關的文法解釋、範例句子以及練習題，幫助學生掌握文法規則。
            """,
            f"{self.modes[1]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入寫作建議模式，學生可以發送他們的英文寫作作品，無論是文章、短文或句子，你會根據內容提供建議。
                你將針對文法、句構、用詞、流暢度等方面進行評價，並提供修改建議以及改進範例。
            """,
            f"{self.modes[2]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入發音反饋模式，給符合學生程度的一句英文句子，讓學生練習說看看。
                當你收到學生的音檔後，根據他們的發音與你提供的句子比較，並針對發音差異提供具體的反饋和改進建議。
            """,
            f"{self.modes[3]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入情境對話模式，這裡會模擬一個日常情境，讓學生練習在特定情境下的對話。
                你會扮演其中一方，並根據情境引導學生進行對話。對話過程中，請注意提供相關的單字、片語和表達方式。
            """,
            f"{self.modes[4]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入學習資源模式，根據自己的學習資訊，提供學生額外的學習資源或建議，可以的話盡量附上網址連結。
                如果自己搜尋到的學習資訊與學生想要的資源差異過大，就和學生說自己暫時沒有合適的相關資源，請學生自己找看看，不要無中生有。
            """,
            f"{self.modes[5]}":"""
                你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
                現在進入設定頁面模式，你提供了設定頁面的連結，讓你的學生可以設定英文程度或通知。
                如果學生不清楚自己的英文程度，幫學生做一下測驗，並評估學生程度在A1-A2、B1-B2，或C1-C2。
            """
        }
        self.mode_reply_text = {
            f"{self.modes[0]}":"""模式：文法教學
                在這個模式下，我們會集中學習英文文法。你可以告訴我你想學的文法點，或者我可以根據你的程度來推薦一些合適的文法知識。
                準備好學習文法了嗎？告訴我你想學習的內容吧！
            """,
            f"{self.modes[1]}":"""模式：寫作建議
                在這個模式下，你可以將你的英文寫作發送給我，我會根據內容給你一些改進建議。
                我會幫你檢查文法、用詞、句構等方面，並提供修改範例來幫助你提升寫作能力。
                準備好了嗎？請發送你的寫作內容，我會幫你看看有什麼可以改進的地方。
            """,
            f"{self.modes[2]}":"""模式：發音反饋
                在這個模式下我會給你一些英文句子，讓你練習說看看。
                請使用LINE的錄音錄下你講的內容，我會幫你聽並給你一些發音建議，準備好的話我們就開始。
            """,
            f"{self.modes[3]}":"""模式：情境對話
                在這個模式下，我們會進行一個模擬情境對話。你可以選擇一個情境，像是餐廳點餐、機場報到、買東西等等。
                你會和我一起練習如何在這些情境下進行英文對話。如果準備好了，告訴我你想練習的情境吧！
            """,
            f"{self.modes[4]}":"""模式：資源推薦
                在這個模式下我可以提供你一些額外的學習資源或建議，讓你參考。
                你需要哪方面的資源呢？像是單字？影片？或是Podcasts？
            """,
            f"{self.modes[5]}":"""模式：設定頁面
                在這個模式下你可以透過設定頁面更改英文程度或通知相關的設定，我會根據你的設定來調整教學方式。
                如果不清楚自己的英文程度，我也可以幫你做一下測驗。
            """
        }
        # 參考CLT認知負荷理論 用在self.modes[0]:文法教學
        self.clt_instruction = """
            教學時應該要舉實際的英文範例給學生參考學習，並且要給學生作練習題目。
            英文學習範例與英文練習題目應該要高度相似，才能讓學生有效學習。
            學生練習後再慢慢變化學習範例，逐步調整練習題目。
            注意要小心在學習與練習中一次出現過多的變量，防止學生一下子學不過來。

            像是學習英文語法時，如果要教的是時間副詞，就要對應多種時間副詞的搭配，而不要同時學習地點副詞、形容詞。

            教學範例如果是：
            昨天我吃了顆蘋果。
            Yesterday, I ate an apple.

            那練習題目應該是讓學生寫：
            (今早/昨晚/剛才/一個半小時前)我吃了顆蘋果。
            (This morning/Last night/Just now/Half an hour ago) I ate an apple.

            然後再逐步變化形式：
            我吃了顆蘋果，(今早/昨晚/剛才/一個半小時前)。
            I ate an apple (this morning/last night/just now/half an hour ago).

            最後再講清楚先時間後動作和先動作後時間的細微差異，之後再使用類似的學習範例加練習題目的組織方式，一步一步學習地點副詞、形容詞、子句等其他知識點。 

            下面請按照這個思路，為學生提供他需要或他想學的知識點的學習範例與相似的練習題目，具體格式如下：
            1) 概念解釋：(說明知識點基本概念)
            2) 學習意義：(為何要學習這個知識點，說明它實際應用的價值)
            3) 學習範例：(提供這個知識點的學習範例)
            4) 範例解析：(解釋這個知識點的學習範例的具體原理與方法)
            5) 練習題目：(提供三個與學習範例高度相關的習題給學生練習)

            學生練習題目時先不要給答案(且題目記得把答案留空)，在學生作答之後，再給出正確答案和指導意見。
        """
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def mode_check(self, user_msg: str) -> bool:
        if match(r'^\*', user_msg):
            for mode in self.modes:
                if user_msg == mode:
                    return True
        return False
    
    def level_check(self, user_msg: str) -> bool:
        if match(r'^\$', user_msg):
            for level in self.levels:
                if user_msg == level:
                    return True
        return False

    def load(self):
        if self.model_loaded:
            print("model is already loaded.")
            return

        llm_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            quantization_config=llm_quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            token=self.hf_token
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_id,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            token=self.hf_token
        )

        self.llm_pipeline = pipeline(
            "text-generation",
            model=llm,
            tokenizer=self.llm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        self.model_loaded = True
        print(f"{id(self)} LLM loaded successfully")

    def unload(self):
        if not self.model_loaded:
            print("model is not loaded.")
            return
        
        # self.db.close()
        
        del self.llm_pipeline
        del self.llm_tokenizer

        torch.cuda.empty_cache()

        self.model_loaded = False
        print(f"{id(self)} LLM has been unloaded successfully.")

    def clear_cache(self):
        torch.cuda.empty_cache()


    def __infer_zh_test(self, prompt: str, prompt_role: str = "學生", system_prompt: str =
        """
           你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
           以中文回應學生。
        """,
        max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 0.6, top_p: float = 0.95) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""
        
        llm_messages = [
            {"role": "系統", "content": f"{system_prompt}"},
            {"role": f"{prompt_role}", "content": f"{prompt}"},
        ]

        llm_prompt = self.llm_tokenizer.apply_chat_template(
            llm_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        llm_prompt = llm_prompt.replace("assistant", "AI英文老師", 1) 
        print(llm_prompt)

        llm_terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        llm_outputs = self.llm_pipeline(
            llm_prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=llm_terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        result = llm_outputs[0]["generated_text"][len(llm_prompt):]

        self.clear_cache()

        return result


    def _infer(self, llm_messages: list[dict[str, str]], generation_prompt_role: str = "AI English teacher", add_generation_prompt: bool = True,
                max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 0.6, top_p: float = 0.95) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""

        llm_prompt = self.llm_tokenizer.apply_chat_template(
            llm_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        if add_generation_prompt:
            llm_prompt = sub(r'assistant(?!.*assistant)', generation_prompt_role, llm_prompt)

        llm_terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        llm_outputs = self.llm_pipeline(
            llm_prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=llm_terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        result = llm_outputs[0]["generated_text"][len(llm_prompt):]

        self.clear_cache()

        return result

    def _infer_zh(self, llm_messages: list[dict[str, str]], generation_prompt_role: str = "AI英文老師", add_generation_prompt: bool = True,
                    max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 0.6, top_p: float = 0.95) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""

        llm_prompt = self.llm_tokenizer.apply_chat_template(
            llm_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        if add_generation_prompt:
            llm_prompt = sub(r'assistant(?!.*assistant)', generation_prompt_role, llm_prompt)

        llm_terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        llm_outputs = self.llm_pipeline(
            llm_prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=llm_terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        result = llm_outputs[0]["generated_text"][len(llm_prompt):]

        self.clear_cache()

        return result
    
    def infer(self, prompt: str, prompt_role: str = "student", generation_prompt_role: str = "AI English teacher",
              system_prompt: str =
        """
            You are an AI English teacher conversing with students on LINE, helping them learn English.
        """
    ) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""

        llm_messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": f"{prompt_role}", "content": f"{prompt}"},
        ]

        result = self._infer(llm_messages, generation_prompt_role)

        self.clear_cache()

        return result
    
    def infer_zh(self, prompt: str, prompt_role: str = "學生", generation_prompt_role: str = "AI英文老師",
                 system_prompt: str =
        """
            你是一位AI英文老師，正和學生們在LINE交談，幫助他們學習英文。
            注意教學生時，盡量多使用中文輔助教學。
        """
    ) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""

        llm_messages = [
            {"role": "系統", "content": f"{system_prompt}"},
            {"role": f"{prompt_role}", "content": f"{prompt}"},
        ]

        result = self._infer_zh(llm_messages, generation_prompt_role)

        self.clear_cache()

        return result

    def infer_with_db(self, user_id: str, prompt: str, prompt_role: str = "student", generation_prompt_role: str = "AI English teacher",
                           add_to_history: str = True, system_prompt: str =
        """
            You are an AI English teacher conversing with students on LINE, helping them learn English.
            Teach student English based on their English level.
        """, rag_infomation: Optional[str] = None, rag_information_process: bool = False
    ) -> str:
        if not self.model_loaded:
            print("LLM is not loaded.")
            return ""
        
        if self.db is None:
            print("RelationalDB is None.")
            return ""
        
        current_mode, english_level = self.db.get_data_by_primary_key('user_settings', user_id, 'current_mode, english_level')
        conversation_history = self.db.get_conversations_by_user(user_id=user_id, max_history_length=self.max_history_length)
        system_prompt = self.system_prompts[current_mode]
        level_description = self.level_descriptions[english_level]

        # conversation_summary = self.abstract(prompt)

        if rag_infomation is not None and rag_information_process:
            rag_infomation = self.infer(prompt=rag_infomation,
                                        prompt_role="retrieved information",
                                        generation_prompt_role="summarization assistant",
                                        add_to_history=False,
                                        system_prompt="""
                                            You are a summarization assistant.
                                            Summarize the key points of the retrieved information below.
                                            The information may be fragmented and may contain overlapping parts due to chunking.
                                            Keep it concise and to the point, removing any redundant or repeated information.
                                            Note! If there are hyperlinks in the information, try to retain them as complete as possible.
                                            
                                            Example:
                                            Retrieved Information:
                                            1. AI teacher introduced themselves and asked for the user's name and English level. The user, John, mentioned their
                                            2. John, mentioned their English level is B1. The AI teacher recommended John to watch English
                                            3. recommended John to watch English pronunciation videos. 
                                            
                                            Summary:
                                            * The AI teacher introduced themselves and asked for the user's name and English level.
                                            * The user, John, mentioned their English level is B1.
                                            * The AI teacher recommended John to watch English pronunciation videos.
                                            
                                            Retrieved Information to summarize:
                                            """
                                        )
                    
        llm_messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "student's english level description", "content": f"{level_description}"},
            # {"role": "conversation summary", "content": f"{conversation_summary}"}
        ]

        # 文法教學
        if current_mode == self.modes[0]:
            llm_messages.append({"role":"instruction", "content":f"{self.clt_instruction}"})
        
        # 資源推薦
        if current_mode == self.modes[4] and rag_infomation is not None:
            llm_messages.append({"role":"learning resource information", "content":f"{rag_infomation}"})

        for speaker, history in conversation_history:
            llm_messages.append({"role": "conversation history", "content": f"{speaker}:{history}"})

        llm_messages.append({"role": f"{prompt_role}", "content": f"{prompt}"})

        print(llm_messages)
        
        result = self._infer(llm_messages, generation_prompt_role)

        if add_to_history:
            conversation_id = str(uuid.uuid4())
            self.db.insert_data('conversation_history',
                                'conversation_id, speaker, conversation_data, timestamp',
                                (conversation_id, prompt_role, prompt, datetime.now(timezone.utc).isoformat()))
            self.db.insert_data('user_conversation_relation', 
                                'user_id, conversation_id', 
                                (user_id, conversation_id))
            
            conversation_id = str(uuid.uuid4())
            self.db.insert_data('conversation_history',
                                'conversation_id, speaker, conversation_data, timestamp',
                                (conversation_id, generation_prompt_role, result, datetime.now(timezone.utc).isoformat()))
            self.db.insert_data('user_conversation_relation', 
                                'user_id, conversation_id', 
                                (user_id, conversation_id))

        self.clear_cache()

        return result

    def abstract(self, prompt: str) -> str:
        conversation_summary = self.infer(prompt=prompt, prompt_role="conversation history", generation_prompt_role="summarization assistant",
                                           add_to_history=False, system_prompt=
                """
                    You are a summarization assistant.
                    Summarize the key points of the conversation below.
                    Keep it concise and to the point.
                    
                    Example:
                    Conversation History:
                    You: Hi, I'm your AI teacher. What's your name and your English level?
                    Student: My name is John. My English level is B1.
                    
                    Summary:
                    * The AI teacher introduced themselves and asked for the user's name and English level.
                    * The user, John, mentioned their English level is B1.
                    
                    Conversation to summarize:
                    """
        )

        # information_extract = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are an information extraction assistant. Extract the key information from the following conversation, focusing on the main questions and responses. Provide a brief summary of the most important points. Keep the summary concise.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Extracted key information:
        #             * Main questions:
        #             + What's your name?
        #             + What's your current English level (roughly)?
        #             * Main responses:
        #             + John is the user's name.
        #             + The user's English level is B1.
                    
        #             Brief summary of the most important points:
        #             The AI teacher introduced themselves and asked for the user's name and English level. The user, John, responded with their name and stated their English level is B1.
                    
        #             Conversation to extract:
        #             """
        # )

        # topic_summary = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are a topic summarization assistant. Identify and summarize the main topics discussed in the following conversation. Provide a brief summary that captures the essence of each topic. Keep it concise and focused.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Topic summary:
        #             * Introduction and Language Level:
        #             + The AI teacher asked the user to introduce themselves and share their English level. The user, John, responded with their name and English level (B1).
                    
        #             Conversation to summarize:
        #             """
        # )

        # context_aware_summary = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are a context-aware summarization assistant. Summarize the conversation, maintaining the context and flow. Ensure the summary captures the flow and connection between different parts of the conversation. Keep it concise.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Context-aware summary:
        #             The conversation started with the AI teacher introducing themselves and asking the user to share their name and current English level. The user, John, responded with their name and English level (B1). This sets the context for the AI teacher to tailor their lessons to John's proficiency level.
                    
        #             Conversation to summarize:
        #             """
        # )

        # print()
        # print(f"conversation_summary: {conversation_summary}")
        # print()
        # print(f"information_extract: {information_extract}")
        # print()
        # print(f"topic_summary: {topic_summary}")
        # print()
        # print(f"context_aware_summary: {context_aware_summary}")
        # print()

        # return (conversation_summary, information_extract, topic_summary, context_aware_summary)
        return conversation_summary
    
    def abstract_zh(self, prompt: str) -> str:
        conversation_summary = self.infer_zh(prompt=prompt, prompt_role="對話紀錄", generation_prompt_role="摘要助手",
                                              add_to_history=False, system_prompt=
                """
                    你是位摘要助手。
                    抓出以下對話紀錄的摘要。
                    注意保持摘要簡潔並切重要點。
                    
                    舉例：
                    對話紀錄：
                    你：嗨，我是你的AI老師。請問你叫什麼名字？以及你目前的英文程度在哪裡呢？
                    學生：我叫John，我的英文程度在B1。
                    
                    摘要：
                    * AI老師做了自我介紹，並問了學生的姓名和英文程度。
                    * 學生叫Jhon，英文程度是B1。
                    
                    待處理的對話紀錄:
                    """
        )

        # conversation_review = self.infer_zh(prompt=prompt, prompt_role="對話紀錄", system_prompt=
        #         """
        #             你是位統整專家。
        #             對話紀錄。
        #             注意保持摘要簡潔並切重要點。
                    
        #             舉例：
        #             對話紀錄：
        #             你：嗨，我是你的AI老師。請問你叫什麼名字？以及你目前的英文程度在哪裡呢？
        #             學生：我叫John，我的英文程度在B1。
                    
        #             摘要：
        #             * AI老師做了自我介紹，並問了學生的姓名和英文程度。
        #             * 學生叫Jhon，英文程度是B1。
                    
        #             待處理的對話紀錄:
        #             """
        # )

        # information_extract = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are an information extraction assistant. Extract the key information from the following conversation, focusing on the main questions and responses. Provide a brief summary of the most important points. Keep the summary concise.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Extracted key information:
        #             * Main questions:
        #             + What's your name?
        #             + What's your current English level (roughly)?
        #             * Main responses:
        #             + John is the user's name.
        #             + The user's English level is B1.
                    
        #             Brief summary of the most important points:
        #             The AI teacher introduced themselves and asked for the user's name and English level. The user, John, responded with their name and stated their English level is B1.
                    
        #             Conversation to extract:
        #             """
        # )

        # topic_summary = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are a topic summarization assistant. Identify and summarize the main topics discussed in the following conversation. Provide a brief summary that captures the essence of each topic. Keep it concise and focused.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Topic summary:
        #             * Introduction and Language Level:
        #             + The AI teacher asked the user to introduce themselves and share their English level. The user, John, responded with their name and English level (B1).
                    
        #             Conversation to summarize:
        #             """
        # )

        # context_aware_summary = self.infer(prompt=prompt, prompt_role="conversation history", system_prompt=
        #         """
        #             You are a context-aware summarization assistant. Summarize the conversation, maintaining the context and flow. Ensure the summary captures the flow and connection between different parts of the conversation. Keep it concise.
                    
        #             Example:
        #             Conversation:
        #             A: Hi, I'm your AI teacher. What's your name and your English level?
        #             B: My name is John. My English level is B1.
                    
        #             Context-aware summary:
        #             The conversation started with the AI teacher introducing themselves and asking the user to share their name and current English level. The user, John, responded with their name and English level (B1). This sets the context for the AI teacher to tailor their lessons to John's proficiency level.
                    
        #             Conversation to summarize:
        #             """
        # )

        # print()
        # print(f"conversation_summary: {conversation_summary}")
        # print()
        # print(f"information_extract: {information_extract}")
        # print()
        # print(f"topic_summary: {topic_summary}")
        # print()
        # print(f"context_aware_summary: {context_aware_summary}")
        # print()

        # return (conversation_summary, information_extract, topic_summary, context_aware_summary)
        return conversation_summary
    
if __name__ == "__main__":
    llm_id = "MaziyarPanahi/Llama-3-8B-Instruct-v0.8"
    cache_dir = "../llm/model"

    llm = LLM(llm_id=llm_id, cache_dir=cache_dir)
    llm.load()

    result = llm.__infer_zh_test("老師的英文怎麼說")
    print(result)