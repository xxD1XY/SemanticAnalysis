import torch
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from torch.nn.functional import softmax
import pandas as pd

@st.cache_resource
def load_models_and_tokenizers():
    sentiment_models = {
        "RuBERT Tiny": ("cointegrated/rubert-tiny-sentiment-balanced", AutoModelForSequenceClassification),
    }
    summarizer_models = {
        "ruT5-base": ("sarahai/ruT5-base-summarizer", AutoModelForSeq2SeqLM),
    }
    sentiment_tokenizers = {name: AutoTokenizer.from_pretrained(model_name) for name, (model_name, _) in sentiment_models.items()}
    summarizer_tokenizers = {name: AutoTokenizer.from_pretrained(model_name) for name, (model_name, _) in summarizer_models.items()}
    sentiment_models = {name: model_class.from_pretrained(model_name, num_labels=3) for name, (model_name, model_class) in sentiment_models.items()}
    summarizer_models = {name: model_class.from_pretrained(model_name) for name, (model_name, model_class) in summarizer_models.items()}

    return sentiment_tokenizers, sentiment_models, summarizer_tokenizers, summarizer_models

def sentiment_analysis(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    result = softmax(outputs.logits, dim=-1).detach().numpy().flatten()
    return result

def summarize_text(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, num_beams=4, max_length=50)
    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summarized_text

def plot_sentiment_analysis(results, labels):
    fig, ax = plt.subplots()
    ax.pie(results, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.legend(labels, title="Легенда", loc="upper right")
    return fig

def main():
    st.title('Semantic Sentiment Analysis')

    st.sidebar.header("Настройки")
    sentiment_tokenizers, sentiment_models, summarizer_tokenizers, summarizer_models = load_models_and_tokenizers()В

    sentiment_model_choice = st.sidebar.selectbox("Выберите модель для анализа тональности", list(sentiment_models.keys()))
    summarizer_model_choice = st.sidebar.selectbox("Выберите модель для суммирования текста", list(summarizer_models.keys()))

    sentiment_tokenizer = sentiment_tokenizers[sentiment_model_choice]
    sentiment_model = sentiment_models[sentiment_model_choice]
    summarizer_tokenizer = summarizer_tokenizers[summarizer_model_choice]
    summarizer_model = summarizer_models[summarizer_model_choice]

    user_input = st.text_area("Введите текст для анализа:", key="user_input")

    if st.button("Анализировать"):
        try:
            # Perform sentiment analysis on the input text
            original_result = sentiment_analysis(user_input, sentiment_tokenizer, sentiment_model)

            # Summarize the input text
            summarization_text = summarize_text(user_input, summarizer_tokenizer, summarizer_model)

            # Perform sentiment analysis on the summarized text
            summarized_result = sentiment_analysis(summarization_text, sentiment_tokenizer, sentiment_model)

            # Display the summarized text
            st.write("Сводка:")
            st.write(summarization_text)

            # Provide a download button for the summary
            st.download_button(
                label="Скачать сводку",
                data=summarization_text,
                file_name="summary.txt",
                mime="text/plain"
            )

            # Visualize the sentiment analysis results
            labels = ['Негативный', 'Нейтральный', 'Позитивный']
            st.write("Результаты анализа тональности исходного текста:")
            st.pyplot(plot_sentiment_analysis(original_result, labels))

            st.write("Результаты анализа тональности суммированного текста:")
            st.pyplot(plot_sentiment_analysis(summarized_result, labels))

            # Display detailed sentiment results in a table
            st.write("Подробные результаты анализа тональности исходного текста:")
            original_df = pd.DataFrame([original_result], columns=labels)
            st.table(original_df)

            st.write("Подробные результаты анализа тональности суммированного текста:")
            summarized_df = pd.DataFrame([summarized_result], columns=labels)
            st.table(summarized_df)

        except Exception as e:
            st.error(f"Ошибка при загрузке модели или выполнении анализа: {e}")

if __name__ == "__main__":
    main()


