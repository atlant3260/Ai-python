from transformers import pipeline
import warnings
import logging

# Полностью отключаем все предупреждения библиотеки transformers, чтобы консоль была чистой
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("запускаем нейросеть")

# Инициализируем генератор текста с русскоязычной моделью
# Первый запуск займет время
generator = pipeline("text-generation", model="ai-forever/rugpt3small_based_on_gpt2")

print("Введи начало истории....\n")
print("для выхода напиши 'выход'\n")

while True:
    # Получаем текст от пользователя
    user_input = input("Твое начало: ")
    
    if user_input.lower() == 'выход': 
        break
        
    print("ИИ думает...")
    
    # Генерируем текст. 
    # Используем max_new_tokens, чтобы ИИ дописал ровно 150 новых токенов.
    # clean_up_tokenization_spaces=False убирает ворнинг токенизатора.
    result = generator(user_input, 
                       max_new_tokens=150, 
                       num_return_sequences=1, 
                       truncation=True, 
                       do_sample=True, 
                       temperature=0.8, 
                       repetition_penalty=1.2,
                       clean_up_tokenization_spaces=False)

    # Вывод результата
    generated_text = result[0]['generated_text']
    print('результат: \n')
    print(generated_text)
    print("\n" + "="*40 + "\n") # Разделитель для удобства чтения