#%%
from copy import copy
import logging
from typing import Any, List, TypeGuard

logging.basicConfig(\
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#from num_token import calc_token_tiktoken

import openai
import os
import streamlit as st
from typing import Any, List
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
PAST_INPUT_MAX_TOKENS = 1024

st.title("StreamlitのChatGPTサンプル")

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
ASSISTANT_WARNING = '注意：私はAIチャットボットで、情報が常に最新または正確であるとは限りません。重要な決定をする前には、他の信頼できる情報源を確認してください。'

def trim_tokens(messages: List[dict], max_tokens: int,
                model_name : str = 'gpt-3.5-turbo-0301')\
                    -> List[dict]:
    total_tokens = calc_token_tiktoken(str(messages),
                                       model_name=model_name)
    while total_tokens > max_tokens:
        messages.pop(0)
        total_tokens = \
            calc_token_tiktoken(str(messages), 
                                model_name=model_name)

            
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
    #logging.info(type(user_msg))
    messages = copy(past_msgs)
    messages.append({"role": "user", "content": user_msg})
    logging.info(f"trim_tokens前のmessages: {messages}")
    logging.info(f"trim_tokens前のmessagesのトークン数: {calc_token_tiktoken(str(messages))}")
    #logging.info(f"trim_tokens前のmessages_type: {type(messages)}")
    
    messages = trim_tokens(messages, PAST_INPUT_MAX_TOKENS)
    logging.info(f"trim_tokens後のmessages: {str(messages)}")
    logging.info(f"trim_tokens後のmessagesのトークン数: {calc_token_tiktoken(str(messages))}")
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        stream=True)
    return response


#%%
with st.chat_message(ASSISTANT_NAME):
        st.write(ASSISTANT_WARNING)

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
    #st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": ASSISTANT_WARNING})

user_msg = st.chat_input("ここにメッセージを入力")
if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])
    user_msg_tokens = calc_token_tiktoken(user_msg)
    if user_msg_tokens > PAST_INPUT_MAX_TOKENS:
        st.warning("メッセージが長すぎます。短くしてください。")
    else:
        # 最新のメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)

        # アシスタントのメッセージを表示
        past_msgs = [{"role": chat["name"],
                      "content": chat["msg"]} \
            for chat in st.session_state.chat_log]
        response = response_chatgpt(user_msg, past_msgs)
        with st.chat_message(ASSISTANT_NAME):
            assistant_msg = ""
            assistant_response_area = st.empty()
            for chunk in response:
                # 回答を逐次表示
                tmp_assistant_msg = chunk["choices"][0]["delta"].get("content", "")
                assistant_msg += tmp_assistant_msg
                assistant_response_area.write(assistant_msg)
        # セッションにチャットログを追加
        st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
        st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})

        logging.info(f"チャットログ: {st.session_state.chat_log}")
    
    
    
#%%
