from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import uuid

from app import init_db
from app.db import SessionLocal
from app.models import Conversation, Message
from app.graph.graph import build_graph

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
graph = build_graph()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    body = request.get_json(force=True) or {}
    user_input = (body.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    conversation_id = (body.get("conversation_id") or "").strip() or uuid.uuid4().hex

    db = SessionLocal()
    try:
        conv = db.get(Conversation, conversation_id)
        if not conv:
            conv = Conversation(id=conversation_id, context={})
            db.add(conv)
            db.commit()
            db.refresh(conv)

        # store user message
        db.add(Message(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
            meta={}
        ))
        db.commit()

        ctx = conv.context or {}
        # Run LangGraph with ctx memory
        state = {
            "conversation_id": conversation_id,
            "user_input": user_input,
            "convo_context": ctx,
        }
        out = graph.invoke(state)

        # Save updated context
        updated_ctx = out.get("updated_context")
        if isinstance(updated_ctx, dict):
            conv.context = updated_ctx
            db.add(conv)
            db.commit()

        assistant_reply = out.get("reply", "") or ""

        db.add(Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_reply,
            meta={
                "trace": out.get("trace", []),
                "results": out.get("results", {}),
                "next_question": out.get("next_question")
            }
        ))
        db.commit()

        return jsonify({
            "conversation_id": conversation_id,
            "reply": assistant_reply,
            "next_question": out.get("next_question"),
            "results": out.get("results", {}),
            "trace": out.get("trace", []),
            "context": conv.context
        })

    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
