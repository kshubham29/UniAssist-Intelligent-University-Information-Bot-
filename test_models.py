import google.generativeai as genai

genai.configure(api_key="AIzaSyACLHGxxkCaGjwMpgI2RLQQJ5hPPsH4-C8")

models = genai.list_models()

for m in models:
    if "generateContent" in m.supported_generation_methods:
        print(m.name)