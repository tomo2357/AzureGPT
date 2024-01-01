#%%
import logging

logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from copy import copy
from typing import Any, List

import openai, os
import streamlit as st
import tiktoken

def calc_token_tiktoken(chat: str, encoding_name: str = "",
                        model_name: str = "gpt-3.5-turbo-0301") -> int:
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    elif model_name:
        encoding = tiktoken.get_encoding(
            tiktoken.encoding_for_model(model_name).name)
    else:
        raise ValueError("Both encoding_name and model_name are missing.")
    num_tokens = len(encoding.encode(chat))
    return num_tokens



# APIキーの設定
openai.api_key = os.environ['OPENAI_API_KEY']

# 過去の最大トークン数
#PAST_INPUT_MAX_TOKENS = 20
PAST_INPUT_MAX_TOKENS = 1024

st.title("StreamlitのChatGPTサンプル")

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
ASSISTANT_WARNING = '注意：私はAIチャットボットで、情報が常に最新または正確であるとは限りません。重要な決定をする前には、他の信頼できる情報源を確認してください。'

def trim_tokens(messages: List[dict], max_tokens: int,
              model_name: str = 'gpt-3.5-turbo-0301') -> List[dict]:
   """
   メッセージのトークン数が指定した最大トークン数を超える場合、
   メッセージの先頭から順に削除し、トークン数を最大トークン数以下に保つ。

   引数:
       messages (List[dict]): メッセージのリスト。
       max_tokens (int): 最大トークン数。
       model_name (str): モデル名（デフォルトは'gpt-3.5-turbo-0301'）。

   戻り値:
       List[dict]: トークン数が最大トークン数以下になったメッセージのリスト。
   """
   # 無限ループを開始
   while True:
       # 現在のメッセージのトークン数を計算
       total_tokens = calc_token_tiktoken(str(messages), model_name=model_name)
       # トークン数が最大トークン数以下になった場合、ループを終了
       if total_tokens <= max_tokens:
           break
       # トークン数が最大トークン数を超えている場合、メッセージの先頭を削除
       messages.pop(0)

   # 修正されたメッセージのリストを返す
   return messages


def response_chatgpt(user_msg: str, past_msgs: List[dict] = [], model_name : str = "gpt-3.5-turbo") -> Any:
    """
    ChatGPTからのレスポンスを取得します。

    引数:
        user_msg (str): ユーザーからのメッセージ。
        past_msgs (List[dict]): 過去のメッセージのリスト。
        model_name (str): 使用するChatGPTのモデル名。デフォルトは"gpt-3.5-turbo"。

    戻り値:
        response: ChatGPTからのレスポンス。
    """
    #logging.debug(type(user_msg))
    messages = copy(past_msgs)
    messages.append({"role": "user", "content": user_msg})
    logging.debug(f"trim_tokens前のmessages: {messages}")
    logging.debug(f"trim_tokens前のmessagesのトークン数: {calc_token_tiktoken(str(messages))}")
    #logging.debug(f"trim_tokens前のmessages_type: {type(messages)}")
    
    messages = trim_tokens(messages, PAST_INPUT_MAX_TOKENS)
    logging.debug(f"trim_tokens後のmessages: {str(messages)}")
    logging.debug(f"trim_tokens後のmessagesのトークン数: {calc_token_tiktoken(str(messages))}")
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        stream=True)
    return response


#%%

# 利用可能なGPTモデルのリスト
available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0301", "その他のモデル"]

# Streamlitサイドバーにモデル選択のドロップダウンを追加
model_choice = st.sidebar.selectbox(
    "GPTモデルを選択してください",
    available_models,
    index=0  # デフォルトの選択肢
)
use_past_data = st.sidebar.checkbox(
    '２つ前以上の会話データを使う', value=True
    )
# アシスタントからの警告を載せる
with st.chat_message(ASSISTANT_NAME):
    st.write(ASSISTANT_WARNING)

# Streamlitアプリの開始時にセッション状態を初期化
if "initialized" not in st.session_state:
    st.session_state.chat_log = []
    st.session_state.initialized = True

# 以前のチャットログを表示
for chat in st.session_state.chat_log:
    with st.chat_message(chat["name"]):
        st.write(chat["msg"])
user_msg = st.chat_input("ここにメッセージを入力")
# 処理開始
if user_msg:
    # 以前のチャットログを表示
    #for chat in st.session_state.chat_log:
    #    with st.chat_message(chat["name"]):
    #        st.write(chat["msg"])
    user_msg_tokens = calc_token_tiktoken(str(
        [
            {
                'role' : 'user', 'content' : user_msg
            }
        ]
        ))
    logging.debug(f'入力したトークン数 : {user_msg_tokens}')
    if user_msg_tokens > PAST_INPUT_MAX_TOKENS:
        st.text_area("入力されたメッセージ", user_msg, height=100)  # メッセージを再表示
        st.warning("メッセージが長すぎます。短くしてください。"
                   f"({user_msg_tokens}tokens)")
        # 処理終了
        #st.session_state.processing = False
    else:
        # 最新のメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)

        # アシスタントのメッセージを表示
        if use_past_data:
            past_msgs = [{"role": chat["name"],
                        "content": chat["msg"]} \
                for chat in st.session_state.chat_log]
        else:
            past_msg = []
        # セッションにチャットログを追加
        st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
        st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": ""})
        response = response_chatgpt(user_msg, past_msgs)
        with st.chat_message(ASSISTANT_NAME):
            assistant_msg = ""
            assistant_response_area = st.empty()
            for chunk in response:
                # 回答を逐次表示
                tmp_assistant_msg = chunk["choices"][0]["delta"].get("content", "")
                assistant_msg += tmp_assistant_msg
                st.session_state.chat_log[-1]["msg"] = assistant_msg
                assistant_response_area.write(assistant_msg)
        # セッションにチャットログを追加
        #st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
        #st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})

        logging.debug(f"チャットログ: {st.session_state.chat_log}")
        logging.debug(f'use_past_data : {use_past_data}')
        #logging.debug(f'use_past_data : {use_past_data}')
        
        # 処理終了
    
#%%
