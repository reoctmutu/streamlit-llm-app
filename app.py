import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# 専門家の振る舞い（A/B）に応じたシステムメッセージ
EXPERT_SYSTEMS = {
	"A": (
		"あなたは旅行プランニングの専門家です。ユーザーの目的、予算、移動手段、"
		"季節や安全面を考慮し、現実的で具体的な旅程案を日本語で提案してください。"
		"可能なら日程ごとのアクティビティ、目安費用、予約のコツも含めてください。"
	),
	"B": (
		"あなたはキャリアコーチの専門家です。ユーザーの目標、経験、スキルギャップを踏まえ、"
		"達成可能なアクションプランを日本語で提案してください。"
		"短期/中期/長期のステップ、学習リソース、ネットワーキングの方法、想定課題と対策を含めてください。"
	),
}


def get_openai_api_key() -> str | None:
	"""環境変数または Streamlit secrets から OpenAI API キーを取得"""
	key = os.getenv("OPENAI_API_KEY")
	if not key:
		try:
			# st.secrets は設定がないと KeyError を投げる可能性あり
			key = st.secrets.get("OPENAI_API_KEY")
		except Exception:
			key = None
	return key


def ask_llm(input_text: str, expert_choice: str) -> str:
	"""
	入力テキストと専門家選択（A/B）を受け取り、LLMの回答テキストを返す。
	"""

	if expert_choice not in EXPERT_SYSTEMS:
		raise ValueError("expert_choice は 'A' または 'B' を指定してください")

	api_key = get_openai_api_key()
	if not api_key:
		raise RuntimeError("OpenAI API キーが見つかりません。環境変数 OPENAI_API_KEY または st.secrets に設定してください。")

	system_msg = EXPERT_SYSTEMS[expert_choice]

	# Lesson8のスタイル: SystemMessage + HumanMessage の会話履歴をLLMへ
	messages = [
		SystemMessage(content=system_msg),
		HumanMessage(content=input_text),
	]

	# ChatOpenAI: Lesson8に準拠しつつ最新APIでinvoke
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
	ai_response = llm.invoke(messages)
	return ai_response.content.strip()


def main():
	st.set_page_config(page_title="LangChain × Streamlit LLMデモ", page_icon="🤖")
	st.title("LangChain × Streamlit LLMデモ")

	# アプリ概要と操作方法
	st.markdown(
		"""
		**概要**
		- このアプリは、入力したテキストを LangChain 経由で LLM に渡し、回答を表示します。
		- ラジオボタンで「専門家A / 専門家B」を選ぶと、LLMのシステムメッセージ（役割）が切り替わります。

		**操作方法**
		- フォームに相談内容（入力テキスト）を記入
		- 専門家種別（A/B）を選択
		- 「送信」ボタンで回答を生成
		"""
	)

	# API キーの案内
	st.info(
		"OpenAI API キーを環境変数 OPENAI_API_KEY または Streamlit の secrets に設定してください。"
	)

	# ラジオボタン（A/B を選択）
	expert_choice = st.radio(
		"専門家の種類を選択",
		options=["A", "B"],
		index=0,
		help="A=旅行プランナー / B=キャリアコーチ"
	)

	# 専門家の説明
	with st.expander("専門家の説明", expanded=False):
		st.write("A: 旅行プランナーの専門家。現実的で安全な旅程を提案します。")
		st.write("B: キャリアコーチの専門家。行動可能なキャリア計画を提案します。")

	# 入力フォーム
	with st.form("llm_input_form", clear_on_submit=False):
		input_text = st.text_area("入力テキスト", placeholder="相談したい内容を具体的に書いてください。", height=150)
		submitted = st.form_submit_button("送信")

	if submitted:
		if not input_text.strip():
			st.warning("入力テキストを入力してください。")
			return

		try:
			response = ask_llm(input_text=input_text.strip(), expert_choice=expert_choice)
			st.success("LLMの回答")
			st.write(response)
		except Exception as e:
			st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
	main()

