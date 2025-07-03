# 🧠 Synapse Engine

> **O Coração Cognitivo do Ecossistema Maestro.AI**
> 
> _Um serviço de memória e raciocínio baseado em um Grafo de Conhecimento Temporal (GraphRAG)._

[![Status do Projeto](https://img.shields.io/badge/status-em_desenvolvimento-yellow)](https://github.com/) [![Versão](https://img.shields.io/badge/versão-v0.1_(Arquitetura)-blue)](./docs/CHANGELOG.md) [![Linguagem](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Arquitetura](https://img.shields.io/badge/arquitetura-GraphRAG-blueviolet)](https://arxiv.org/abs/2404.18731) [![API](https://img.shields.io/badge/api-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/) [![Licença](https://img.shields.io/badge/licença-pendente-lightgrey)](./LICENSE)

## 🧠 O Que é o Synapse Engine?

O **Synapse Engine** não é um banco de dados vetorial. É o **sistema nervoso central cognitivo** para o ecossistema `Maestro.AI`. Sua missão é ir além da simples recuperação de informações (RAG tradicional) para fornecer aos agentes de IA uma **memória organizacional profunda, contextual e historicamente consciente**.

Em vez de armazenar documentos isolados, o Synapse Engine constrói um **Grafo de Conhecimento (Knowledge Graph)** dinâmico a partir dos projetos. Ele não apenas sabe _o que_ está no código, mas entende _como_ as funções se conectam, _por que_ uma decisão foi tomada e _quando_ uma mudança crucial ocorreu.

Esta capacidade de raciocínio sobre relações e tempo é o que permite aos nossos agentes tomar decisões inteligentes e evitar os erros de contexto comuns em sistemas de IA mais simples.

### ✨ As Quatro Forças Nucleares

O poder do Synapse Engine reside em quatro capacidades fundamentais:

1. **Fluência Estrutural e Semântica:** Compreende tanto a estrutura sintática do código (via ASTs) quanto a intenção semântica por trás dele (via LLMs).
    
2. **Consciência Temporal:** Trata cada `commit` como um evento no tempo, permitindo "viajar no tempo" para entender a evolução e a causa raiz dos problemas.
    
3. **Raciocínio Inferencial Multi-Salto:** Permite aos agentes "ligar os pontos" e descobrir relações complexas e indiretas que seriam invisíveis de outra forma.
    
4. **Motor da Governança:** Fornece os atributos contextuais em tempo real que alimentam o motor de políticas (ABAC) do `Maestro.AI`, transformando regras estáticas em julgamento dinâmico.
    

## 🏗️ Arquitetura de Alto Nível (GraphRAG)

O Synapse Engine implementa uma arquitetura **GraphRAG (Retrieval-Augmented Generation sobre Grafos de Conhecimento)**, que é superior ao RAG vetorial para tarefas complexas de engenharia de software.

```
graph LR
    subgraph Fontes de Dados
        A[Repositórios Git]
        B[Documentação do Codex Prime]
        C[APIs Externas]
    end

    subgraph Pipeline de Ingestão
        D[Análise Estrutural<br>(e.g., tree-sitter)]
        E[Análise Semântica<br>(LLM)]
    end

    subgraph Synapse Engine Core
        F[(Grafo de Conhecimento<br>Temporal)]
        G[API de Consulta (GraphQL/REST)]
    end

    subgraph Consumidores
        H[Maestro.AI]
        I[Agentes de IA]
    end

    A --> D;
    B --> E;
    C --> E;
    D --> F;
    E --> F;
    F --> G;
    G --> H;
    G --> I;
```

1. **Ingestão:** Um pipeline automatizado analisa continuamente as fontes de dados, extraindo entidades (funções, classes, issues) e relações (`CHAMA`, `IMPLEMENTA`, `RESOLVE`).
    
2. **Construção do Grafo:** As informações extraídas são usadas para construir e enriquecer dinamicamente o Grafo de Conhecimento central, onde cada mudança é versionada por `commit`.
    
3. **Consulta:** O `Maestro.AI` e os agentes fazem perguntas complexas ao Synapse Engine através de sua API, recebendo como resposta não apenas texto, mas subgrafos de conhecimento rico e contextual.
    

## 🔌 API e Endpoints Principais

O Synapse Engine expõe sua funcionalidade através de uma API RESTful (construída com FastAPI).

- `POST /ingest`: Aciona o pipeline de ingestão para um novo repositório ou fonte de dados.
    
- `POST /query`: O endpoint principal para consulta. Aceita uma pergunta em linguagem natural e retorna um subgrafo de conhecimento relevante.
    
- `GET /entity/{id}`: Recupera informações detalhadas sobre uma entidade específica no grafo (e.g., uma função ou um agente).
    
- `GET /history/{entity_id}`: Retorna o histórico de versões de uma entidade específica, permitindo a análise temporal.
    

## 🚀 Como Começar (Getting Started)

Instruções para configurar e executar uma instância local do `Synapse Engine`.

### Pré-requisitos

- Python 3.11+
    
- Docker e Docker Compose
    
- Uma instância de uma base de dados de grafos (e.g., Neo4j, recomendada para o início)
    

### Instalação

1. **Clone o repositório:**
    
    ```
    git clone [URL_DO_REPOSITORIO_SYNAPSE]
    cd SynapseEngine
    ```
    
2. Inicie os serviços com Docker Compose:
    
    Este comando irá iniciar a API do Synapse e o banco de dados de grafos.
    
    ```
    docker-compose up -d
    ```
    
3. **Faça sua primeira ingestão:**
    
    ```
    curl -X POST http://localhost:8000/ingest \
         -H "Content-Type: application/json" \
         -d '{"source_type": "git", "uri": "https://github.com/exemplo/repositorio.git"}'
    ```
    
4. **Faça sua primeira consulta:**
    
    ```
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "Qual a função principal do módulo de autenticação?"}'
    ```
    

## 📖 Documentação

A documentação arquitetural completa do `Synapse Engine` pode ser encontrada em [`/docs/03_TECNOLOGIA_ENGINEERING/SYNAPSE_ENGINE_HLD.md`](https://gemini.google.com/app/docs/03_TECNOLOGIA_ENGINEERING/SYNAPSE_ENGINE_HLD.md "null") dentro da instância do `Codex Prime` deste projeto.

## 🤝 Como Contribuir

O `Synapse Engine` é um projeto complexo e de vanguarda. As contribuições mais valiosas estão nas áreas de:

- **Melhorar os Parsers de Código:** Adicionar suporte para novas linguagens de programação no pipeline de ingestão.
    
- **Otimizar Consultas de Grafo:** Desenvolver algoritmos de consulta mais eficientes.
    
- **Pesquisar Novos Modelos de Grafo:** Explorar novas formas de modelar o conhecimento de software.
    

Por favor, consulte o nosso [guia de contribuição](https://gemini.google.com/app/CONTRIBUTING.md "null") para mais detalhes.

## 📄 Licença

Distribuído sob a licença [Nome da Licença]. Veja `LICENSE.txt` para mais informações.

## 📞 Contato

Maestro (Desenvolvedor Principal): Bruno S. Rosa

LinkedIn: [/In/BrunoSRosa](https://linkedin.com/in/brunosrosa)