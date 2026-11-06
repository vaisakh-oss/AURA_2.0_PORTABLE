MIRA 2.0 — Agentic Multimodal RAG System

An AI-powered document understanding and reasoning assistant designed to retrieve information from documents, process user queries, and support structured responses using an agent-based architecture.

Overview

MIRA 2.0 is an Agentic Retrieval-Augmented Generation (RAG) system that combines document retrieval, language-model reasoning, and tool execution.

Unlike a basic question-answering system, the project is designed to use an agent workflow to interpret user queries, retrieve relevant information, and use available tools to generate context-aware responses.

Problem Statement

Searching through large documents manually can be time-consuming. Traditional question-answering systems may also struggle when information is distributed across multiple documents or requires multiple reasoning steps.

MIRA 2.0 aims to address these challenges through document retrieval, hybrid search, and an agent-based execution workflow.

Key Features

- Agent-Based Reasoning: Uses a ReAct-style execution workflow for handling multi-step queries.
- Hybrid Retrieval: Combines semantic vector search with BM25 keyword-based retrieval.
- Document Understanding: Designed to extract and retrieve information from supported documents.
- Tool Integration: Uses a tool registry to make available functions accessible to the agent.
- Persistent Storage: Includes application memory and storage components.
- Evaluation Framework: Includes evaluation-related modules and test files for assessing system behaviour.
- Security Considerations: Includes security documentation and security-focused tests.

System Architecture

The project is organised into the following major components:

Component| Responsibility
User Interface| Accepts user queries and displays responses
Agent| Coordinates the reasoning workflow
Planner| Supports task planning
Knowledge Retrieval| Retrieves relevant document information
Hybrid Search| Combines semantic and keyword retrieval
Tools| Provides functions the agent can use
Memory| Supports information persistence
Evaluation| Supports testing and evaluation

Retrieval Approach

The hybrid retrieval approach combines:

1. Semantic Search: Uses sentence-transformer embeddings to identify contextually relevant information.
2. BM25 Search: Uses keyword matching to retrieve documents based on relevant terms.
3. Reciprocal Rank Fusion (RRF): Combines the rankings from different retrieval methods.

Technology Stack

- Python
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- BM25
- Agent-based reasoning
- Document processing
- Automated testing

Project Structure

AURA_2.0_PORTABLE/
│
├── agent.py
├── app.py
├── planner.py
├── tools.py
├── knowledge.py
├── knowledge_store.py
├── document_extractor.py
├── memory.py
├── storage.py
├── classifier.py
├── vision.py
├── evaluation.py
├── analytics.py
│
├── tests and verification files
├── setup scripts
├── start scripts
├── README.md
├── SECURITY.md
└── requirements.txt

Note: This is a simplified representation of the main components, not the complete directory listing.

Installation and Execution

The repository includes setup and startup scripts for supported environments.

1. Clone or download the repository.
2. Open the project directory.
3. Review the requirements and configuration instructions.
4. Configure the required API credentials using the provided environment template.
5. Run the appropriate setup script for your operating system.
6. Follow the startup instructions provided with the project.

Important: Refer to the setup scripts and configuration files for the exact commands and dependencies.

Testing and Evaluation

The repository includes test modules covering different areas of the application, including:

- Agent behaviour
- Knowledge retrieval
- Memory handling
- Application functionality
- Integration workflows
- Security checks
- Evaluation workflows

The presence of these tests does not imply that every test currently passes. Test results should be verified before release.

Security

Sensitive configuration and runtime data should not be committed to the public repository.

Refer to "SECURITY.md" (SECURITY.md) for security-related information.

Project Information

Project: MIRA 2.0 — Agentic Multimodal RAG System
Domain: Artificial Intelligence and Robotics
Focus Areas: Agentic AI, RAG, Document Understanding and Information Retrieval

---

This project is developed as an academic engineering project. Its features and performance should be evaluated against the actual implementation and test results.
