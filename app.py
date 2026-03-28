import base64
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Експертне голосування", layout="wide")

OBJECTS = [
    "Sirius", "Black Hole", "Mercury", "Venus", "Andromeda",
    "Earth", "Mars", "Wormhole", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "Moon", "Europa",
    "Titan", "Milky Way", "Callisto", "Sun", "Comet",
]

HEURISTICS = {
    "E1": "Об'єкт обирався 1 раз на 3-му місці",
    "E2": "Об'єкт обирався 1 раз на 2-му місці",
    "E3": "Об'єкт обирався 1 раз  на 1-му місці",
    "E4": "Об'єкт обирався 2 рази на 3-му місці",
    "E5": "Об'єкт обирався 2 рази, один раз на 3-му і один раз на 2-му місці",
    "E6": "Сума балів <= 3",
    "E7": "Об'єкт жодного разу не обирався на 1-му місці",
}

VOTES_FILE = "votes.csv"
H_VOTES_FILE = "heuristic_votes.csv"
ADMIN_PASSWORD = "admin123"

SEED_H_VOTES = [
    ("Вiка", "E7", "E6", "E1"),
    ("Анна", "E6", "E7", "E2"),
    ("Іван","E7", "E1", "E4"),
    ("Ромчик", "E6", "E4", "E5"),
    ("Анастасiя", "E7", "E5", "E6"),
    ("Лiза", "E1", "E6", "E7"),
    ("Валерiя", "E6", "E7", "E3"),
    ("Лера", "E7", "E2", "E6"),
    ("Даша", "E6", "E1", "E7"),
    ("Настя","E7", "E6", "E4"),
    ("Максим", "E6", "E5", "E7"),
    ("Веронiка","E7", "E6", "E1"),
    ("Вiкторiя","E6", "E7", "E5"),
    ("Дарина", "E7", "E4", "E6"),
    ("Марина", "E6", "E7", "E2"),
    ("Anastasiia", "E7", "E6", "E3"),
    ("Михайло","E6", "E1", "E7"),
    ("Дарiя","E7", "E6", "E5"),
    ("хтось", "E6", "E7", "E4"),
    ("Оскар", "E7", "E5", "E6"),
]

def init_h_votes_file() -> None:
    if not os.path.exists(H_VOTES_FILE):
        df_seed = pd.DataFrame(SEED_H_VOTES, columns=["name", "h1", "h2", "h3"])
        df_seed.to_csv(H_VOTES_FILE, index=False)

init_h_votes_file()

def set_bg(gif_path: str) -> None:
    if not os.path.exists(gif_path):
        return
    with open(gif_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/gif;base64,{b64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0,0,0,.68);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 2rem;
        }}
        h1,h2,h3,h4,h5,h6 {{ color: #00FFFF; }}
        .stButton>button {{
            background: linear-gradient(90deg,#4B0082,#00008B);
            color: white; font-weight: bold;
            border-radius: 10px; transition: all .3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 6px #8A2BE2, 0 0 22px #1E90FF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg("images/starfield.gif")

def load_scores():
    # зчитує votes.csv і обчислює бали
    scores = {o: 0 for o in OBJECTS}
    counts = {o: {"c1": 0, "c2": 0, "c3": 0} for o in OBJECTS}
    if not os.path.exists(VOTES_FILE):
        return scores, counts
    df = pd.read_csv(VOTES_FILE)
    for _, row in df.iterrows():
        for col, pts, key in [("choice1", 3, "c1"), ("choice2", 2, "c2"), ("choice3", 1, "c3")]:
            obj = str(row.get(col, "")).strip()
            if obj in scores:
                scores[obj] += pts
                counts[obj][key] += 1
    return scores, counts


def goodfor_heuristic(obj: str, key: str, counts: dict, scores: dict) -> bool:
    # повертає True, якщо об'єкт підпадає під евристику відсіювання
    c = counts[obj]
    total = c["c1"] + c["c2"] + c["c3"]
    if key == "E1":
        return total == 1 and c["c3"] == 1
    if key == "E2":
        return total == 1 and c["c2"] == 1
    if key == "E3":
        return total == 1 and c["c1"] == 1
    if key == "E4":
        return total == 2 and c["c3"] == 2
    if key == "E5":
        return total == 2 and c["c3"] == 1 and c["c2"] == 1 and c["c1"] == 0
    if key == "E6":
        return scores[obj] <= 3
    if key == "E7":
        return c["c1"] == 0
    return False

def load_h_votes() -> pd.DataFrame:
    if not os.path.exists(H_VOTES_FILE):
        return pd.DataFrame(columns=["name", "h1", "h2", "h3"])
    return pd.read_csv(H_VOTES_FILE)

def ranked_heuristics_from_votes(df_h: pd.DataFrame) -> list[tuple[str, int]]:
    # повертає список (евристика, бали) за спаданням
    h_scores = {k: 0 for k in HEURISTICS}
    for _, row in df_h.iterrows():
        for col, pts in [("h1", 3), ("h2", 2), ("h3", 1)]:
            k = str(row.get(col, "")).strip()
            if k in h_scores:
                h_scores[k] += pts
    return sorted(h_scores.items(), key=lambda x: -x[1])


def apply_heuristicsStep(objects_list: list, heuristics_order: list, counts: dict, scores: dict) -> tuple[list, list]:
    # послідовно застосовує евристики, поки залишилось > 10 об'єктів
    current = list(objects_list)
    log = []
    for h_key in heuristics_order:
        if len(current) <= 10:
            break
        removed = [o for o in current if goodfor_heuristic(o, h_key, counts, scores)]
        if removed:
            current = [o for o in current if o not in removed]
        log.append({
            "Евристика": h_key,
            "Опис": HEURISTICS[h_key],
            "Видалено": ", ".join(removed) if removed else "—",
            "Залишилось": len(current),
        })
    return current, log

def genetic_rank(
    objects_subset: list,
    scores: dict,
    pop_size: int = 60,
    generations: int = 200,
    mut_rate: float = 0.10,
) -> tuple[list, list]:
    n = len(objects_subset)
    if n == 0:
        return [], []

    def fitness(perm: list) -> float:
        return sum(scores.get(o, 0) * (n - rank) for rank, o in enumerate(perm))

    def crosover(p1: list, p2: list) -> list:
        a, b = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[a : b + 1] = p1[a : b + 1]
        fill = [x for x in p2 if x not in child]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[j]
                j += 1
        return child

    def mutate(perm: list) -> list:
        p = perm[:]
        for i in range(n):
            if random.random() < mut_rate:
                j = random.randint(0, n - 1)
                p[i], p[j] = p[j], p[i]
        return p

    popul = [random.sample(objects_subset, n) for _ in range(pop_size)]
    best_perm = None
    best_fit = float("-inf")
    history = []

    for _ in range(generations):
        ranked_pop = sorted(popul, key=fitness, reverse=True)
        top_fit = fitness(ranked_pop[0])
        if top_fit > best_fit:
            best_fit = top_fit
            best_perm = ranked_pop[0][:]
        history.append(best_fit)

        survivors = ranked_pop[: pop_size // 2]
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(crosover(p1, p2))
            new_pop.append(child)
        popul = new_pop

    return best_perm, history

scores, counts = load_scores()

tab = st.sidebar.selectbox(
    "Розділ",
    [
        "Результати ЛР1",
        "Голосування за евристики",
        "Застосування евристик",
        "Генетичний алгоритм",
        "Адмін",
    ],
)

if tab == "Результати ЛР1":
    st.title("Результати лабораторної роботи №1")
    st.markdown(
        "Рейтинг за підсумками голосування"
    )

    rows = []
    for o in OBJECTS:
        rows.append(
            {
                "Об'єкт": o,
                "1-е місце": counts[o]["c1"],
                "2-е місце": counts[o]["c2"],
                "3-є місце": counts[o]["c3"],
                "Загалом обрано раз": counts[o]["c1"] + counts[o]["c2"] + counts[o]["c3"],
                "Сума балів": scores[o],
            }
        )
    df_res = (
        pd.DataFrame(rows)
        .sort_values("Сума балів", ascending=False)
        .reset_index(drop=True)
    )
    df_res.index += 1
    st.dataframe(df_res, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    objs_sorted = df_res["Об'єкт"].tolist()
    vals = df_res["Сума балів"].tolist()
    ax.bar(objs_sorted, vals, color="white")
    ax.set_xlabel("Об'єкт", color="white")
    ax.set_ylabel("Сума балів", color="white")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors="white", axis="both", labelrotation=45)
    for sp in ax.spines.values():
        sp.set_color("white")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.pyplot(fig)

elif tab == "Голосування за евристики":
    st.title("Голосування за пріоритетність евристик")

    st.subheader("Перелік евристик")
    for k, v in HEURISTICS.items():
        st.markdown(f"**{k}** — {v}")

    st.divider()
    name = st.text_input("Ваше ім'я")
    h_keys = list(HEURISTICS.keys())
    h1 = st.selectbox("1-й пріоритет", h_keys, key="h1")
    h2 = st.selectbox("2-й пріоритет", h_keys, key="h2")
    h3 = st.selectbox("3-й пріоритет", h_keys, key="h3")
    if st.button("Проголосувати"):
        if not name.strip():
            st.error("Введіть ім'я")
        elif len({h1, h2, h3}) < 3:
            st.error("Оберіть 3 різні евристики")
        else:
            df_h = load_h_votes()
            new_row = pd.DataFrame(
                [[name.strip(), h1, h2, h3]], columns=["name", "h1", "h2", "h3"]
            )
            df_h = pd.concat([df_h, new_row], ignore_index=True)
            df_h.to_csv(H_VOTES_FILE, index=False)
            st.success(f"Голос збережено. Ваш вибір: **{h1}** > **{h2}** > **{h3}**")

elif tab == "Застосування евристик":
    st.title("Застосування евристик")
    df_h = load_h_votes()
    if len(df_h) == 0:
        st.warning(
            "Ще немає голосів за евристики. "
            "Перейдіть у вкладку **Голосування за евристики** та проголосуйте"
        )
        st.stop()

    ranked = ranked_heuristics_from_votes(df_h)
    st.subheader("Ранжування евристик за підсумками голосування")
    h_rank_df = pd.DataFrame(
        [{"Евристика": k, "Опис": HEURISTICS[k], "Сума балів": v} for k, v in ranked]
    )
    st.dataframe(h_rank_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4.5, 2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ks = [r[0] for r in ranked]
    vs = [r[1] for r in ranked]
    ax.bar(ks, vs, color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.set_xlabel("Евристика", color="white")
    ax.set_ylabel("Бали", color="white")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.pyplot(fig)

    st.divider()
    st.subheader("Покрокове застосування евристик")
    ordered_keys = [k for k, _ in ranked]
    final_set, step_log = apply_heuristicsStep(OBJECTS, ordered_keys, counts, scores)

    st.dataframe(pd.DataFrame(step_log), use_container_width=True)
    st.subheader(f"Фінальна підмножина: {len(final_set)} об'єктів")
    final_df = pd.DataFrame(
        [
            {
                "Об'єкт": o,
                "1-е місце": counts[o]["c1"],
                "2-е місце": counts[o]["c2"],
                "3-є місце": counts[o]["c3"],
                "Сума балів": scores[o],
            }
            for o in final_set
        ]
    ).sort_values("Сума балів", ascending=False).reset_index(drop=True)
    final_df.index += 1
    st.dataframe(final_df, use_container_width=True)

    if len(final_set) <= 10:
        st.success(f"Підмножину успішно звужено до **{len(final_set)} об'єктів**")
    else:
        st.warning(
            f"Залишилось **{len(final_set)}** об'єктів"
            "Зберіть більше голосів або застосуйте додаткові евристики."
        )

elif tab == "Генетичний алгоритм":
    st.title("Генетичний алгоритм")
    df_h = load_h_votes()
    if len(df_h) == 0:
        st.warning("Немає голосів за евристики, використовується порядок E1...E7 за замовчуванням")
        ordered_keys = list(HEURISTICS.keys())
    else:
        ranked = ranked_heuristics_from_votes(df_h)
        ordered_keys = [k for k, _ in ranked]

    final_set, _ = apply_heuristicsStep(OBJECTS, ordered_keys, counts, scores)

    st.info(
        f"Підмножина для ранжування: **{len(final_set)} об'єктів**: "
        + ", ".join(final_set)
    )

    pop_size = 80
    generations = 200
    mut_rate = 0.10
    if st.button("Запустити ГА"):
        with st.spinner("Виконується генетичний алгоритм.."):
            best_perm, history = genetic_rank(
                final_set, scores, pop_size, generations, mut_rate
            )

        st.subheader("Результат ранжування")
        ga_df = pd.DataFrame(
            [
                {"Місце": i + 1, "Об'єкт": o, "Бали (ЛР1)": scores[o]}
                for i, o in enumerate(best_perm)
            ]
        )
        st.dataframe(ga_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(4.5, max(2, len(best_perm) * 0.22)))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        reversed_perm = list(reversed(best_perm))
        vals = [scores[o] for o in reversed_perm]
        ax.barh(
            reversed_perm,
            vals,
            color="white",
        )
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.set_xlabel("Бали", color="white")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.pyplot(fig)

        st.success(f"Переможець: **{best_perm[0]}**  ({scores[best_perm[0]]} балів ЛР1)")

elif tab == "Адмін":
    st.title("Адміністративна панель")
    password = st.text_input("Пароль", type="password")

    if password == ADMIN_PASSWORD:
        st.success("Доступ надано")

        st.subheader("Протокол голосування за евристики")
        df_h = load_h_votes()
        if len(df_h):
            st.dataframe(df_h, use_container_width=True)
            with open(H_VOTES_FILE, "rb") as fh:
                st.download_button(
                    "Завантажити протокол евристик",
                    fh,
                    "heuristic_votes.csv",
                    "text/csv",
                )
        else:
            st.info("Голосів за евристики ще немає.")

        if st.button("Очистити голоси за евристики"):
            pd.DataFrame(columns=["name", "h1", "h2", "h3"]).to_csv(
                H_VOTES_FILE, index=False
            )
            st.success("Голоси за евристики видалено.")

        st.divider()

        st.subheader("Протокол голосування ЛР1 (votes.csv)")
        if os.path.exists(VOTES_FILE):
            df_v = pd.read_csv(VOTES_FILE)
            st.dataframe(df_v, use_container_width=True)
            with open(VOTES_FILE, "rb") as fh:
                st.download_button(
                    "Завантажити votes.csv",
                    fh,
                    "votes.csv",
                    "text/csv",
                )
        else:
            st.info("Файл votes.csv не знайдено.")
    elif password:
        st.error("Невірний пароль")