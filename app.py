import streamlit as st
import pickle
import numpy as np
import html

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Dark Navy Styling
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #020617 0%, #071426 45%, #0b1f3a 100%);
    color: white;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.hero-card {
    background: linear-gradient(135deg, #0f2a4d, #071426);
    padding: 35px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}

.stat-card {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
}

.stat-number {
    font-size: 28px;
    font-weight: 800;
    color: #38bdf8;
}

.stat-label {
    font-size: 14px;
    color: #cbd5e1;
}

.section-title {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 30px;
    margin-bottom: 20px;
}

.book-card {
    background: linear-gradient(180deg, #102a43, #061426);
    padding: 18px;
    border-radius: 20px;
    min-height: 430px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    text-align: center;
    transition: transform 0.2s ease;
}

.book-card:hover {
    transform: translateY(-5px);
}

.book-img {
    width: 150px;
    height: 220px;
    object-fit: cover;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.45);
}

.book-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    min-height: 65px;
}

.book-author {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 5px;
}

.book-info {
    font-size: 14px;
    color: #e2e8f0;
    margin-top: 8px;
}

.recommend-box {
    background: linear-gradient(135deg, #0f2a4d, #061426);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.14);
    margin-top: 20px;
    margin-bottom: 30px;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border-radius: 14px;
    padding: 12px 30px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
}

div[data-baseweb="select"] {
    background-color: white;
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 15px;
}

hr {
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Files
# -----------------------------
try:
    popular_df = pickle.load(open("popular.pkl", "rb"))
    pt = pickle.load(open("pivot.pkl", "rb"))
    similarity_scores = pickle.load(open("similarity.pkl", "rb"))
    books = pickle.load(open("books_small.pkl", "rb"))

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def get_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return col
    return None


title_col = get_column(books, ["Book-Title", "book_title", "title", "Title"])
author_col = get_column(books, ["Book-Author", "book_author", "author", "Author"])
image_col = get_column(books, ["Image-URL-M", "Image-URL-L", "image_url", "Image", "image"])

DEFAULT_IMAGE = "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=400&q=80"


def get_book_details(book_name):
    author = "Unknown Author"
    image = DEFAULT_IMAGE

    try:
        if title_col is not None:
            book_data = books[books[title_col] == book_name]

            if len(book_data) > 0:
                if author_col is not None:
                    author_value = book_data.iloc[0][author_col]
                    if str(author_value) != "nan":
                        author = author_value

                if image_col is not None:
                    image_value = book_data.iloc[0][image_col]
                    if str(image_value) != "nan" and str(image_value).startswith("http"):
                        image = image_value

    except:
        pass

    return author, image


def get_popular_book_title(index, row):
    if "Book-Title" in popular_df.columns:
        return row["Book-Title"]
    return index


def get_rating(row):
    if "avg_rating" in popular_df.columns:
        return round(row["avg_rating"], 2)
    return "N/A"


def get_rating_count(row):
    if "num_rating" in popular_df.columns:
        return row["num_rating"]
    return "N/A"


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(book_name):
    try:
        index = np.where(pt.index == book_name)[0][0]

        similar_books = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        recommendations = []

        for i in similar_books:
            recommendations.append(pt.index[i[0]])

        return recommendations

    except Exception as e:
        st.error(f"Recommendation Error: {e}")
        return []


# -----------------------------
# Hero Section
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("""
    <div class="hero-card">
        <div class="main-title">📚 Book Recommendation System</div>
        <div class="sub-title">
            Discover your next favorite book using AI-powered collaborative filtering.
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.image(
        "https://images.unsplash.com/photo-1512820790803-83ca734da794?auto=format&fit=crop&w=800&q=80",
        use_container_width=True
    )

# -----------------------------
# Stats Section
# -----------------------------
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{pt.shape[0]}</div>
        <div class="stat-label">Books in Model</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{pt.shape[1]}</div>
        <div class="stat-label">Active Users</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">AI</div>
        <div class="stat-label">Recommendation Engine</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["🏆 Popular Books", "🔍 Recommend Books", "📊 Dataset Info"]
)

# -----------------------------
# Popular Books Tab
# -----------------------------
with tab1:
    st.markdown('<div class="section-title">⭐ Top Recommended Books</div>', unsafe_allow_html=True)

    popular_books = popular_df.head(10)

    rows = [popular_books.iloc[i:i+5] for i in range(0, len(popular_books), 5)]

    for row_group in rows:
        cols = st.columns(5)

        for col, (idx, row) in zip(cols, row_group.iterrows()):
            book_name = get_popular_book_title(idx, row)
            author, image = get_book_details(book_name)
            rating = get_rating(row)
            count = get_rating_count(row)

            safe_book = html.escape(str(book_name))
            safe_author = html.escape(str(author))

            with col:
                st.markdown(f"""
                <div class="book-card">
                    <img src="{image}" class="book-img">
                    <div class="book-title">{safe_book}</div>
                    <div class="book-author">{safe_author}</div>
                    <div class="book-info">⭐ Rating: {rating}</div>
                    <div class="book-info">👥 {count} ratings</div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# Recommendation Tab
# -----------------------------
with tab2:
    st.markdown('<div class="section-title">📖 Get Book Recommendations</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="recommend-box">
        <h3 style="color:white;">Select a book you like</h3>
        <p style="color:#cbd5e1;">The system will suggest 5 similar books based on user rating patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    selected_book = st.selectbox(
        "Choose a Book",
        pt.index.values
    )

    if st.button("🔍 Recommend Books"):

        recommended_books = recommend(selected_book)

        if len(recommended_books) > 0:

            st.markdown('<div class="section-title">❤️ Books You May Like</div>', unsafe_allow_html=True)

            cols = st.columns(5)

            for col, book in zip(cols, recommended_books):
                author, image = get_book_details(book)

                safe_book = html.escape(str(book))
                safe_author = html.escape(str(author))

                with col:
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{image}" class="book-img">
                        <div class="book-title">{safe_book}</div>
                        <div class="book-author">{safe_author}</div>
                        <div class="book-info">Recommended for you</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.warning("No recommendations found.")

# -----------------------------
# Dataset Info Tab
# -----------------------------
with tab3:
    st.markdown('<div class="section-title">📊 Dataset Information</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{popular_df.shape}</div>
            <div class="stat-label">Popular Books Data</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{pt.shape}</div>
            <div class="stat-label">Pivot Table Shape</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{similarity_scores.shape}</div>
            <div class="stat-label">Similarity Matrix Shape</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("View Raw Popular Books Data"):
        st.dataframe(popular_df.head(20))
