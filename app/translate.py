import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

def anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

def system_template():
    return "Translate the following from English into {language}"

def prompt_template(sys_template):
    return ChatPromptTemplate.from_messages(
        [("system", sys_template), ("user", "{text}")]
    )

def language_prompt(langs):
    print("Welcome to a simple llm tutorial!\n")
    print("Please pick a language to translate to.\n")
    for index, lang in enumerate(langs):
        print(f"[{index}] {lang}")

    user_selection=int(input(">"))
    print(f"You have selected {langs[user_selection]}!\n")

    return user_selection

def display_model_response(prompt, model):
    if os.environ.get("DEBUG_TOKEN").lower() == 'true':
        print("\n###################")
        print("### TOKEN DEBUG ###")
        print("###################\n")
        for token in model.stream(prompt):
            print(token.content, end="|")
    else:
        response = model.invoke(prompt)
        print(response.content)

def translate():
    anthropic_key()
    lang_list = ["Italian", "Spanish", "German", "French"]
    user_lang = language_prompt(lang_list)
    user_text = input("What would you like to translate?\n>")
    prompt = prompt_template(system_template()).invoke({"language": lang_list[user_lang], "text": user_text})
    model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
    prompt = prompt_template(system_template()).invoke({"language": lang_list[user_lang], "text": user_text})
    display_model_response(prompt, model)

translate()