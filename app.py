import streamlit as st

from groq import Groq

from supabase import create_client, Client

import os

from dotenv import load_dotenv

from rag_utils import extract_text_from_pdf, chunk_text, create_embedding, find_relevant_chunks


# Load environment variables

load_dotenv(override=True)


# Set up the page

st.set_page_config(page_title="Knowledge Assistant", page_icon="🤖", layout="wide")


# Initialize clients

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

supabase: Client = create_client(

    os.getenv("SUPABASE_URL"),

    os.getenv("SUPABASE_KEY")

)


# Sidebar

with st.sidebar:

    st.header("📚 Document Management")

    

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    

    if uploaded_file:

        if st.button("📤 Process Document", use_container_width=True):

            with st.spinner("Processing document..."):

                try:
                    text = extract_text_from_pdf(uploaded_file)

                    st.success(f"✅ Extracted {len(text)} characters")
                    with st.expander("🔍 DEBUG: View Extracted Text (first 1000 chars)"):
                        st.code(text[:1000])

                    chunks = chunk_text(text)

                    st.info(f"📦 Created {len(chunks)} chunks")
                    with st.expander("🔍 DEBUG: View First 3 Chunks"):
                        for i, chunk in enumerate(chunks[:3]):
                            st.write(f"**Chunk {i+1}:**")
                            st.code(chunk[:300])

                    doc_response = supabase.table("documents").insert({

                        "user_id": "default_user",

                        "filename": uploaded_file.name,

                        "file_size": uploaded_file.size,

                        "total_chunks": len(chunks)

                    }).execute()

                    doc_id = doc_response.data[0]['id']

                    progress_bar = st.progress(0)

                    for idx, chunk in enumerate(chunks):

                        embedding = create_embedding(chunk)

                        supabase.table("document_chunks").insert({

                            "document_id": doc_id,

                            "chunk_index": idx,

                            "chunk_text": chunk,

                            "embedding": embedding

                        }).execute()

                        progress_bar.progress((idx + 1) / len(chunks))

                    st.success(f"✅ Processed {len(chunks)} chunks from {uploaded_file.name}")

                    st.rerun()

                except Exception as e:

                    st.error(f"Error: {str(e)}")

    

    st.divider()

    

    st.subheader("📄 Your Documents")

    try:

        docs = supabase.table("documents").select("*").eq("user_id", "default_user").order("upload_date", desc=True).execute()

        

        if docs.data:

            for doc in docs.data:

                with st.expander(f"📎 {doc['filename']}"):

                    st.caption(f"Chunks: {doc['total_chunks']}")

                    st.caption(f"Size: {doc['file_size'] / 1024:.1f} KB")

                    if st.button("🗑️ Delete", key=f"del_{doc['id']}"):

                        supabase.table("documents").delete().eq("id", doc["id"]).execute()

                        st.rerun()

        else:

            st.info("No documents uploaded yet")

    except Exception as e:

        st.error(f"Error loading documents: {e}")

    

    st.divider()

    

    st.header("💬 Chat Controls")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        supabase.table("chat_messages").delete().eq("user_id", "default_user").execute()

        st.session_state.messages = []

        st.rerun()

    

    try:

        msg_count = supabase.table("chat_messages").select("*", count="exact").eq("user_id", "default_user").execute()

        doc_count = supabase.table("documents").select("*", count="exact").eq("user_id", "default_user").execute()

        st.metric("Messages", msg_count.count)

        st.metric("Documents", doc_count.count)

    except Exception:

        pass


# Main chat area

st.title("🤖 Knowledge Assistant")

st.caption("Ask questions about your uploaded documents!")


if "messages" not in st.session_state:

    try:

        response = supabase.table("chat_messages").select("*").eq("user_id", "default_user").order("created_at").execute()

        st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in response.data]

    except Exception:

        st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


user_input = st.chat_input("Ask about your documents...")


if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):

        st.write(user_input)

    

    try:

        supabase.table("chat_messages").insert({

            "user_id": "default_user",

            "role": "user",

            "content": user_input

        }).execute()

    except Exception:

        pass

    

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                chunks_response = supabase.table("document_chunks").select("chunk_text, embedding, document_id, chunk_index").execute()

                context = ""

                if chunks_response.data:

                    relevant = find_relevant_chunks(user_input, chunks_response.data, top_k=3)

                    context = "\n\n".join(

                        [

                            f"[Document excerpt {i + 1}]: {chunk['chunk_text'][:300]}..."

                            for i, chunk in enumerate(relevant)

                        ]

                    )

                    enhanced_messages = st.session_state.messages[:-1] + [

                        {

                            "role": "user",

                            "content": f"""Based on these document excerpts:



{context}



Question: {user_input}



Please answer the question using the information from the documents. If the documents don't contain relevant information, say so.""",

                        }

                    ]

                else:

                    enhanced_messages = st.session_state.messages

                

                response = groq_client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=enhanced_messages,

                    max_tokens=1000,

                    temperature=0.7

                )

                

                assistant_response = response.choices[0].message.content

                

                st.write(assistant_response)

                

                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                supabase.table("chat_messages").insert({

                    "user_id": "default_user",

                    "role": "assistant",

                    "content": assistant_response

                }).execute()

                

            except Exception as e:

                st.error(f"Error: {str(e)}")
