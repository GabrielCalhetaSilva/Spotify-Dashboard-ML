#Importações(Streamlit biblioteca que permite criar e compartilhar aplicativos web interativos/Pandas pra manipulação de dados/plotly para gráfico interativos/sklearn para o Machine Learning(ML)

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#Configurações, definindo o título da aba com ícone e um layout mais largo e bonito

st.set_page_config(
    page_title="Spotify Dashboard",
    page_icon="🎧",
    layout="wide"
)

#Estilo(CSS), markdown permite escrever HTML/CSS no Streamlit, optei por um fundo escuro com títulos brancos e padding 15px por border 10px nas caixinhas ali os KPIs 

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3 {
        color: #1DB954;
    }
    .stMetric {
        background-color: #181818;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

#Leitura do csv

df = pd.read_csv('data/top2018.csv')

#Barra lateral a Sidebar, com campo de busca, coloquei o case=False para ignorar maiúscula/minúscula, e filtros de energia e dançabilidade dos sons, que são os tópicos mais interessantes na minha opinião

st.sidebar.title("Filtros")

search = st.sidebar.text_input("🔍 Buscar música")

if search:
    df = df[df['name'].str.contains(search, case=False)]

energy = st.sidebar.slider(
    "Energia",
    float(df['energy'].min()),
    float(df['energy'].max()),
    (0.0, 1.0)
)
df = df[(df['energy'] >= energy[0]) & (df['energy'] <= energy[1])]

dance = st.sidebar.slider(
    "Dançabilidade",
    float(df['danceability'].min()),
    float(df['danceability'].max()),
    (0.0, 1.0)
)
df = df[(df['danceability'] >= dance[0]) & (df['danceability'] <= dance[1])]

#Header(Título principal)

st.title("🎧 Spotify Dashboard")
st.markdown("Padrões das 100 músicas mais populares do Spotify")

# KPIs, dividi a tela em 3 colunas para mostrar quantidade de músicas e a média de energia e dançabilidade

col1, col2, col3 = st.columns(3)

col1.metric("Músicas", len(df))
col2.metric("Energia média", round(df['energy'].mean(), 2))
col3.metric("Dançabilidade média", round(df['danceability'].mean(), 2))

#Scatter plot, gráfico de dispersão que mostra o X = o quanto a música é dançante e Y = o quanto a música é energética/traz emoções e quando passar o mouse no pontinho você pode ver o nome da música, renderizado e com fundo escuro

st.subheader("Energia X Dançabilidade")

fig = px.scatter(
    df,
    x='danceability',
    y='energy',
    hover_data=['name', 'artists'],
    color='energy',
    size='danceability',
    color_continuous_scale='greens'
)

fig.update_layout(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

#Histogramas

col1, col2 = st.columns(2)

with col1:
    fig2 = px.histogram(df, x='energy', nbins=20, color_discrete_sequence=['#1DB954'])
    fig2.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.histogram(df, x='danceability', nbins=20, color_discrete_sequence=['#1DB954'])
    fig3.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font_color="white")
    st.plotly_chart(fig3, use_container_width=True)

#Top músicas, ordenei as 10 mais dançantes com o head(10) e fiz uma tabela interativa simples

st.subheader("Top 10 mais dançantes")

top = df.sort_values(by='danceability', ascending=False).head(10)

st.dataframe(
    top[['name', 'artists', 'danceability', 'energy']],
    use_container_width=True
)

#Insights que anotei no Notebook

st.subheader("Principais Insights")

st.markdown("""
-  Músicas mais energéticas tendem a ser mais altas
-  Músicas acústicas são menos energéticas
-  Músicas mais dançantes tendem a ser mais positivas
-  Rap e trap tendem a ser mais dançantes desse top 100
-  BPM não define se a música é dançante
""")

#Machine Learning simples de recomendação, basicamente, essa parte do código cria um sistema simples de recomendação de músicas: primeiro o usuário escolhe uma música em um dropdown, e o programa usa algumas características dela — como energia, dançabilidade, valência(felicidade), tempo e nível acústico para representar cada música como um conjunto de números. Em seguida, esses dados são normalizados para que nenhuma característica pese mais que a outra na comparação, e então é calculada a similaridade entre todas as músicas usando uma técnica chamada cosine similarity(ele calcula o quão semelhantes dois vetores são, comparando o ângulo entre eles), que mede o quão “parecidas” elas são. A partir disso, quando o usuário seleciona uma música, o código encontra quais outras têm o perfil mais próximo e retorna as 5 mais similares (ignorando a própria música escolhida), exibindo essas recomendações na tela

st.subheader("Recomendação de Música")

music_list = df['name'].values
selected_music = st.selectbox("Escolha uma música", music_list)

features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

similarity = cosine_similarity(X_scaled)

def recommend(song_name):
    index = df[df['name'] == song_name].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]]['name'] for i in sorted_distances[1:6]]
    return recommended

if selected_music:
    recs = recommend(selected_music)
    
    st.write("Se você gostou dessa música, vai curtir:")
    for r in recs:
        st.markdown(f"- {r}")

#Textinho final

st.markdown("---")
st.markdown("Obrigado por ver até aqui!")
st.markdown("Feito por Gabriel Calheta")