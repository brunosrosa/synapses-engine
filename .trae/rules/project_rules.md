```
---
# METADADOS DO PROJETO (Para uso do Maestro.AI)
project_name: 'Synapse Engine'
codex_framework_version: '3.0' # Baseado na versão do Codex Prime Framework
tech_lead: 'Bruno S. Rosa'
status: 'ativo'
document_version: '3.0'
repository_url: 'https://github.com/brunosrosa/synapse-engine'
---
```

# Regras do Projeto: Synapse Engine

## 1. Diretrizes Gerais e Metodologia

- **Objetivo Principal:** Desenvolver o `Synapse Engine`, uma **Plataforma de Memória Cognitiva Temporal** de alta performance. O sistema implementará uma arquitetura avançada de **GraphRAG**, servindo como o "cérebro" de conhecimento para todo o ecossistema.
    
- **Metodologia:** Desenvolvimento Focado em API e Infraestrutura, seguindo os princípios de "Desenvolvimento Solo Ágil Aumentado por IA". O foco é em design robusto, testes rigorosos (funcionais, de carga, de contrato), e documentação de API impecável.
    
- **Repositório Principal:** `https://github.com/brunosrosa/synapse-engine`
    
- **Caminho da Documentação:** A documentação viva deste projeto é uma instância do `Codex Prime Framework` e reside na pasta `./docs/` na raiz deste repositório.
    

## 2. Stack de Tecnologia e Ferramentas

- **Linguagem Principal (Backend):** Python 3.11+
    
- **Framework Principal (Backend):** FastAPI
    
- **Servidor ASGI:** Uvicorn (com `uvicorn[standard]`) para desenvolvimento e produção.
    
- **Banco de Dados de Grafos:** Neo4j
    
- **Banco de Dados Vetorial:** PostgreSQL (com a extensão `pgvector`)
    
- **Frameworks de IA/NLP:** LlamaIndex (para ingestão e indexação), LangChain (para orquestração de cadeias complexas), Transformers (Hugging Face) (para acesso a modelos de embedding).
    
- **Plataforma de Deploy Alvo:** Docker em VPS (Ubuntu).
    
- **Ferramentas de Qualidade (Python):** Pytest, Black, Flake8, MyPy.
    

## 3. Padrões de Código e Versionamento

- **Convenção de Nomenclatura:**
    
    - **Arquivos de Documentação:** `MAIUSCULA_COM_UNDERLINE`
        
    - **Arquivos de Código Python:** `snake_case` (ex: `rag_service.py`)
        
- **Estilo de Commits:** Utilizar estritamente o padrão de **Conventional Commits** (ex: `feat:`, `fix:`, `perf:`, `refactor:`) para manter um histórico claro.
    
- **Guia de Estilo de Código:** Aderir rigorosamente ao guia de estilo **PEP 8**. A responsabilidade de conhecer e aplicar o padrão é do agente executor.
    

## 4. Hierarquia e Delegação de Agentes

- **Agente Estratégico (Chefe de Gabinete):** `@Janus`
    
- **Agente Tático (Gerente de Projetos):** `@Orquestrador`
    
- **Agentes Especialistas Principais:**
    
    - `@DevPython_FastAPI` (Desenvolvimento da API e da lógica de serviço)
        
    - `@Engenheiro_de_IA_RAG` (Lógica do RAG, embeddings, chunking, busca híbrida)
        
    - `@Arquiteto_de_Dados_Grafo` (Design do schema do Neo4j, otimização de queries Cypher)
        
    - `@Arquiteto_de_Cloud_e_DevOps` (Infraestrutura, Docker, CI/CD)
        

## 5. Protocolo de Resolução de Conflitos

Em caso de conflito ou ambiguidade entre diferentes fontes de regras, a seguinte hierarquia de prioridade **DEVE** ser seguida:

1. **Decisão explícita e atual do Maestro:** Uma instrução direta de Bruno S. Rosa sempre tem a maior prioridade.
    
2. **`project_rules.md` (Este Documento):** As regras específicas para o projeto `Synapse Engine`.
    
3. **`user_rules.md`:** As regras e preferências globais do Maestro.
    
4. **`Regras_do_Jogo.md`:** A constituição geral do ecossistema.
    
5. **Documentação Técnica Viva do Projeto:** Decisões formalizadas em ADRs, HLDs e outros documentos na pasta `./docs/`.
    
6. **Padrões Gerais da Indústria e da Linguagem.**
    
    
O agente que identificar um conflito tem a responsabilidade de sinalizá-lo e solicitar orientação para garantir o alinhamento.