import streamlit as st
import torch
import numpy as np
from src.llm_engine import GPT2Engine
import src.visualizer as visualizer
import html

st.set_page_config(page_title="LLM Autoregressive Explorer", layout="wide")


def update_sampling():
    sampled_probs = engine.apply_sampling(
        st.session_state.logits,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p
    )

    top_words, top_probs = engine.get_top_k_words(
        sampled_probs,
        k=st.session_state.n_words
    )

    st.session_state.top_words = top_words
    st.session_state.top_probs = top_probs


@st.cache_resource
def load_engine():
    return GPT2Engine()


engine = load_engine()

# Initialize session state if it does not exist
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "The black cat sleeps over"

st.title("Autoregressive LLM Explorer")
st.markdown(
    "Step through what happens under the hood of a model like GPT-2 during text generation."
)

# Fix INPUT BUG: Use 'key' to bind it to session_state
current_prompt = st.text_input("📝 Input Phrase (Prompt):", key="prompt_input")

if current_prompt:

    # Run main computations
    input_ids, tokens = engine.tokenize(current_prompt)
    embeddings = engine.get_embeddings(input_ids)

    st.divider()

    # ---------------------------------------------------------
    # PHASE 1: Tokenization
    # ---------------------------------------------------------
    st.header("1. Tokenization")
    st.markdown("Text is split into tokens (subwords) and converted into static numerical IDs from the vocabulary.")

    # 20 soft, modern colors for tokens (pastel optimized for Light Theme)
    colors = [
        "#fca5a5","#fdba74","#fcd34d","#bef264","#86efac","#6ee7b7",
        "#5eead4","#67e8f9","#7dd3fc","#93c5fd","#a5b4fc","#c4b5fd",
        "#d8b4fe","#f0abfc","#f9a8d4","#fda4af","#e7e5e4","#d4d4d8",
        "#e4e4e7","#cbd5e1"
    ]

    tokens_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">'
    for tok, tid in zip(tokens, input_ids[0].tolist()):
        bg_color = colors[tid % len(colors)]
        safe_tok = html.escape(tok)
        tokens_html += f'''
<div style="background-color: {bg_color}; color: #0f172a; padding: 6px 12px; 
            border-radius: 8px; font-family: monospace; font-size: 16px; 
            border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
    <span style="font-weight: 600;">{safe_tok}</span> 
    <span style="font-size: 12px; opacity: 0.7; margin-left: 4px;">({tid})</span>
</div>
'''
    tokens_html += '</div>'

    st.markdown(tokens_html, unsafe_allow_html=True)

    st.divider()

    # ---------------------------------------------------------
    # PHASE 2: Embedding
    # ---------------------------------------------------------
    st.header("2. Embedding")
    st.markdown("Each numerical ID becomes a high-dimensional vector. Here you see a 3D projection.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Vectors (Raw Data)")
        # FIX BUG: Limit height and add native Streamlit scrolling
        with st.container(height=500):
            for i, tok in enumerate(tokens):
                emb_preview = np.round(embeddings[i][:4], 3)
                st.markdown(
                    f"**`{tok}`**<br><span style='color: gray; font-family: monospace; font-size: 14px;'>[{emb_preview[0]}, {emb_preview[1]}, {emb_preview[2]}, {emb_preview[3]}...]</span>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<hr style='margin:4px 0; border:0; height:1px; background:#555;'>",
                    unsafe_allow_html=True
                )

    with col2:
        # Ensure plot has similar height
        fig_emb = visualizer.plot_3d_embeddings(tokens, embeddings)
        fig_emb.update_layout(height=500)
        st.plotly_chart(fig_emb, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # PHASE 3: Transformer and Attention
    # ---------------------------------------------------------
    st.header("3. Contextualization (Self-Attention)")
    st.markdown("""
    See how tokens relate to each other. The top-right is black due to the **Causal Mask** 
    (GPT cannot look at the future). The first token is often very bright (**Attention Sink**).
    *Tip: Middle layers (e.g., 5 or 6) often capture grammar better than the last layers!*
    """)

    logits, attentions = engine.forward_pass(input_ids)
    st.session_state.logits = logits

    # Controls to explore Layer and Heads
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    att_col1, att_col2 = st.columns(2)
    with att_col1:
        layer_idx = st.slider("Select Layer", 0, num_layers - 1, num_layers // 2)
    with att_col2:
        head_idx = st.slider("Select Head (Attention Head)", 0, num_heads - 1, 0)

    # Extract specific 2D matrix (seq_len, seq_len)
    specific_attention_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()

    fig_att = visualizer.plot_attention_heatmap(
        tokens, specific_attention_matrix,
        title_suffix=f"(Layer {layer_idx}, Head {head_idx})"
    )
    st.plotly_chart(fig_att, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # PHASE 4 & 5: Logits, Softmax, Sampling and Generation
    # ---------------------------------------------------------
    st.header("4. Probabilities and Sampling")
    st.markdown(
        "Adjust the parameters to see how the **next word** probability distribution changes."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        n_words = st.slider("Number of words (N)", min_value=5, max_value=100, value=20, step=5)
    with ctrl2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    with ctrl3:
        top_p = st.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

    # Dynamic sampling computation
    sampled_probs = engine.apply_sampling(logits, temperature=temperature, top_p=top_p)
    top_words, top_probs = engine.get_top_k_words(sampled_probs, k=n_words)

    fig_probs = visualizer.plot_probabilities(
        top_words,
        top_probs,
        title=f"Top {n_words} Words after Sampling"
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    st.divider()

    # --- PHASE 5 INCLUDED HERE ---
    st.header("5. Iterative Generation")

    # Dynamically choose token based on new sampling
    if temperature == 0.0:
        chosen_token_idx = np.argmax(sampled_probs)
    else:
        chosen_token_idx = torch.multinomial(torch.tensor(sampled_probs), 1).item()

    chosen_word = engine.tokenizer.decode([chosen_token_idx])
    current_text = st.session_state.prompt_input

    st.markdown(f"""
<div style=" padding:16px; border-radius:10px; background:#f8fafc; border:1px solid #cbd5e1; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">

<div style="font-size:18px; color:#0f172a; margin-bottom:2px;">
Model selected:
<span style="background:#dbeafe; padding:2px 10px; border-radius:6px; font-weight:bold; color:#1e40af; margin-left:6px;"> {chosen_word} </span>
<span style="margin-left:5px; color:#64748b; font-size:14px;"> (ID {chosen_token_idx}) </span>
</div>

<div style=" margin-top:10px; padding:12px; background:#ffffff; border-radius:8px; border:1px solid #e2e8f0; font-family:monospace; font-size:16px; color:#1e293b;">
{current_text}
<span style="color:#0284c7; font-weight:bold; background:#e0f2fe; padding:0 4px; border-radius:4px;">{chosen_word}</span>
</div>
</div>
    """, unsafe_allow_html=True)


    # Callback to handle text update
    def append_word_to_prompt(word):
        st.session_state.prompt_input += word


    st.write("")
    st.button(
        "Next Word",
        on_click=append_word_to_prompt,
        args=(chosen_word,),
        type="primary"
    )

    max_autopilot_steps = 50


    def run_autopilot():
        # Immediately add chosen word and update
        st.session_state.prompt_input += chosen_word
        curr_text = st.session_state.prompt_input

        for _ in range(max_autopilot_steps):
            input_ids, _ = engine.tokenize(curr_text)
            new_logits, _ = engine.forward_pass(input_ids)
            s_probs = engine.apply_sampling(new_logits, temperature=temperature, top_p=top_p)

            if temperature == 0.0:
                c_token_idx = int(np.argmax(s_probs))
            else:
                c_token_idx = torch.multinomial(torch.tensor(s_probs), 1).item()

            next_word = engine.tokenizer.decode([c_token_idx])
            curr_text += next_word
            st.session_state.prompt_input = curr_text

            if next_word.strip() in [".", "!", "?"] or "\n" in next_word:
                break


    st.button(
        "Autopilot",
        on_click=run_autopilot,
        type="secondary"
    )