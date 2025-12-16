import fastchat.model

# Check available templates
template = fastchat.model.get_conversation_template("llama-3")
print(template)

template = fastchat.model.get_conversation_template("qwen-7b-chat")
print(template)

# Check what attributes exist
import fastchat.model
conv = fastchat.model.get_conversation_template("llama-3")
print(dir(conv))
print(conv.__dict__)