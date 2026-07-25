package org.example.Test5;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.example.Assistant;

import java.util.List;
import java.util.Scanner;

public class Test5PasDeRag {

    public static void main(String[] args) {

        String geminiKey = System.getenv("GEMINI_KEY");
        if (geminiKey == null || geminiKey.isBlank()) {
            System.err.println("Environment variable GEMINI_KEY is missing.");
            return;
        }

        // Création du modèle Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequests(true)
                .logResponses(true)
                .build();

        // Parser et ingestion du document RAG
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document docRag = FileSystemDocumentLoader.loadDocument("src/main/resources/rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(512, 25);
        List<TextSegment> segmentsRag = splitter.split(docRag);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        var embeddingsRag = embeddingModel.embedAll(segmentsRag).content();

        EmbeddingStore<TextSegment> storeRag = new InMemoryEmbeddingStore<>();
        storeRag.addAll(embeddingsRag, segmentsRag);

        ContentRetriever retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRag)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        QueryRouter router = query -> {
            String prompt = String.format(
                    "Est-ce que la requête '%s' porte sur l'intelligence artificielle (IA), les LLM ou le RAG ? Réponds uniquement par 'oui', 'non' ou 'peut-être'.",
                    query
            );

            String answer = model.chat(prompt).trim().toLowerCase();
            String firstWord = answer.split("\\s+")[0]; // premier mot

            System.out.println("[ROUTAGE] Réponse du LM : " + firstWord);

            if (firstWord.equals("oui") || firstWord.equals("peut-être")) {
                return List.of(retrieverRag);
            } else {
                return List.of(); // pas de RAG
            }
        };

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(chatMemory)
                .build();

        // Boucle interactive
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) continue;
                if ("fin".equalsIgnoreCase(question)) break;

                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
