from pathlib import Path
import faiss
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig

class TextEmbeddingModel:
    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        # # ensure the path is absolute
        # if not self.cache_dir.is_absolute():
        #     self.cache_dir = self.cache_dir.resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self):
        if self.model_loaded:
            print("model is already loaded.")
            return

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir)
        )

        self.model_loaded = True
    
    def average_pool(self, last_hidden_states: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embedding(self, text: str) -> torch.Tensor:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Please call `load` method first.")
        
        # Tokenize the input text
        batch_dict = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        # Calculate embeddings
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class RAG:
    def __init__(self, embedding_model: TextEmbeddingModel, index_dir: str, text_dir: str):
        import os
        os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.text_dir = Path(text_dir)
        self.index = None
        self.chunks = []
        self.chunked_embeddings = []

        # # ensure the path is absolute
        # if not self.index_dir.is_absolute():
        #     self.index_dir = self.index_dir.resolve()
        # if not self.text_dir.is_absolute():
        #     self.text_dir = self.text_dir.resolve()

        # ensure the directories exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
    
    def new_index(self, index_name: str):
        d = self.embedding_model.get_embedding("test").shape[1]  # Get embedding dimension
        index = faiss.IndexFlatL2(d)
        index_path = self.index_dir / index_name
        faiss.write_index(index, str(index_path))
    
    def read_index(self, index_name: str):
        index_path = self.index_dir / index_name

        if not index_path.exists():
            self.new_index(index_name)

        self.index = faiss.read_index(str(index_path))

    def chunk(self, text: str, chunk_size: int = 128, chunk_overlap: int = 16) -> list:
        tokens = self.embedding_model.tokenizer(text, return_tensors='pt', truncation=False, padding=False)['input_ids'].squeeze().tolist()

        chunks = []
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) < chunk_size:
                break
            chunks.append(self.embedding_model.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        return chunks

    def save_chunks_as_text(self, text_name: str, chunks: list, append: bool = True):
        text_path = self.text_dir / text_name
        
        # Determine file mode based on append parameter
        file_mode = 'a' if append and text_path.exists() else 'w'
        
        # print(f"Writing to: {text_path} (append mode: {append})")
        
        with open(text_path, file_mode, encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + '\n')

    def read_text(self, text_name: str):
        text_path = self.text_dir / text_name
        
        if not text_path.exists():
            self.new_text(index_name)

        with open(text_path, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip() for line in f]

    def save_chunk_as_vector(self, index_name: str, chunk: str):
        if self.index is None:
            raise RuntimeError("Index not loaded. Please call `read_index` method first.")
        
        embeddings = self.embedding_model.get_embedding(chunk)

        if embeddings.is_cuda:
            embeddings = embeddings.cpu()

        embeddings_np = embeddings.numpy()

        self.index.add(embeddings_np)

        index_path = self.index_dir / index_name
        faiss.write_index(self.index, str(index_path))

    def save_chunks_as_vector(self, index_name: str, chunks: list[str]):
        # if not isinstance(chunks, list):
        #     raise ValueError("Expected a list of texts.")
        if len(chunks) == 0:
            raise RuntimeError("The chunks is empty!")
        
        if self.index is None:
            raise RuntimeError("Index not loaded. Please call `read_index` method first.")

        embeddings_list = []
        for chunk in chunks:
            embeddings = self.embedding_model.get_embedding(chunk)
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            embeddings_np = embeddings.numpy()
            embeddings_list.append(embeddings_np)

        embeddings_np = np.vstack(embeddings_list)

        self.index.add(embeddings_np)

        index_path = self.index_dir / index_name
        faiss.write_index(self.index, str(index_path))

    def retrieve(self, query: str, result_num: int = 3) -> list[str]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Please call `read_index` method first.")
        
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Please add vectors to the index first.")
        
        
        query_embedding = self.embedding_model.get_embedding(query)

        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy()

        # distance and index
        D, I = self.index.search(query_embedding_np, k=result_num)
        # print(f"Distance: {D}\nIndex: {I}")

        results = [self.chunks[idx] for idx in I[0]]
        return results

# usually use only two functions below from outside
    def retrieve_by_id(self, id: str, query: str, result_num: int = 3) -> str:
        index_name = f'{id}.index'
        text_name = f'{id}.txt'

        self.read_index(index_name)
        self.read_text(text_name)

        if self.index.ntotal == 0:
            # tmp
            if len(self.chunks) > 0:
                self.save_chunks_as_vector(index_name, self.chunks)
            else:
                # print("Index is empty.")
                # raise RuntimeError("Index is empty. Please add vectors to the index first.")
                return None


        result = ""
        results = self.retrieve(query, result_num=result_num)

        # if results is like ["result_1", "result_2", "result_3"]
        # then the formatted_results is like ["1. result_1", "2. result_2", "3. result_3"]
        formatted_results = []
        for i, r in enumerate(results, start=1):
            formatted_result = f"{i}. {r}"
            formatted_results.append(formatted_result)

        result = '\n'.join(formatted_results)

        return result

    def save_content_by_id(self, id: str, content: list):
        if len(content) == 0:
            return
        
        print("text needs to chunk:")
        print("\n".join(content))

        chunks = self.chunk("\n".join(content))
        print(f"chunks: {chunks}")

        index_name = f'{id}.index'
        text_name = f'{id}.txt'

        self.read_index(index_name)
        self.save_chunks_as_vector(index_name, chunks)
        self.save_chunks_as_text(text_name, content, append=True)
        

if __name__ == "__main__":
    model_id = 'intfloat/multilingual-e5-large-instruct'
    cache_dir = './models'
    embedding_model = TextEmbeddingModel(model_id, cache_dir)
    embedding_model.load()

    text = "Here is an example text to embed."
    embedding = embedding_model.get_embedding(text)
    print(f"Embedding: {embedding}")

    # Initialize RAG
    index_dir = './vector_db/indices'
    text_dir = './vector_db/texts'
    rag = RAG(embedding_model, index_dir, text_dir)

    index_name = 'test_1.index'
    # rag.new_index(index_name)
    # rag.read_index(index_name)

    result = rag.retrieve_by_id('test_1', "test")
    print(result)

    # Load text and add vectors
    # text_name = 'text.txt'
    
    # rag.chunk_and_save_text(text_name, """
    #         A1-A2(零基礎-初學階段)
    #         這時學生還不太會英文，請多用中文進行教學，並建議學生先注重發音與單字的學習，推薦學生看看網路上英文發音的教學影片，並搜尋英文最常用的一千個單字，用自己的方式背看看

    #         你的主要目標：
    #         給學生學習的方向(發音與單字)，幫助學生先背基本的一千個英文單字，以及學習基本的美式英語的發音

    #         重要注意事項：
    #         多使用中文教學

    #         B1-B2(中階-中高階段)
    #         這時學生有一點英文基礎，教學時可以嘗試少用一點中文，依照學生學習喜好，可以建議學生再背更多單字，多聽多看適合自己程度的英文影片，或者找一對一的英文老師練習多使用英文(找你就是個不錯的選擇)，可以用主題式的聊天或閒聊，讓老師再與學生英文交談時，指出學生英文使用上的問題以及如何用得更加流暢

    #         你的主要目標：
    #         讓學生摸索自己的學習方式(偏向自學或與人互動)，嘗試增加學生英文交談能力，點出學生使用英文時的問題，以及改善的方式
                   
    #         C1-C2(精通階段)
    #         這時學生英文基本上很厲害了，基本上都用英文和學生溝通也沒問題。學生可能是再更加貼近母語人士，或是學習特定專業領域的英文，可以先了解學生的學習動機，再考慮教學方向。

    #         主要目標：
    #         了解學生的學習動機，以貼近母語人士，或特定專業領域人士的英文來精進學生的英文能力
    # """)

    # rag.read_text(text_name)
    # rag.add_vector(rag.chunked_embeddings)

    # # Retrieve texts
    # user_input = input("User: ")
    # while user_input != "q":
    #     results = rag.retrieve(user_input)
    #     print("Retrieved texts: ")
    #     for result in results:
    #         print(result)
        
    #     user_input = input("User: ")