import fastchat.model

# Check available templates
template = fastchat.model.get_conversation_template("llama-3")
print(template)

template = fastchat.model.get_conversation_template("qwen-7b-chat")
print(template)