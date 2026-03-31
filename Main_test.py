from Voice_module import conversation_loop, text_to_speech_mode, speech_to_text_mode

def get_backend_response(user_text):
    # Temporary simulation
    if "admission" in user_text:
        return "Admission requires sixty percent in twelfth grade."
    elif "courses" in user_text:
        return "We offer BTech, MBA, and MCA programs."
    else:
        return "This query will be handled by the UniAssist knowledge base."


print("\n=== UniAssist Voice System ===")
print("1. Full Voice Assistant")
print("2. Text Question → Voice Answer")
print("3. Voice Question → Text Answer")

choice = input("Select mode: ")

if choice == "1":
    conversation_loop(get_backend_response)

elif choice == "2":
    text_to_speech_mode(get_backend_response)

elif choice == "3":
    speech_to_text_mode(get_backend_response)

else:
    print("Invalid choice")