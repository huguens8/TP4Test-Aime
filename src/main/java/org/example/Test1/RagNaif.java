package org.example.Test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.example.Assistant;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        // Phase 1 RAG
        String documentPath = "rag.pdf";

        String geminiKey = System.getenv("GEMINI_KEY");

        if (geminiKey == null || geminiKey.isBlank()) {
            System.err.println("Environment variable GEMINI_KEY is missing.");
            return;
        }

        // Creation du modele
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey) // cle de connexion
                .modelName("gemini-2.5-flash") // model utilise
                .temperature(0.2) // temperature
                .build();

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(documentPath, parser);

        DocumentSplitter documentSplitter = DocumentSplitters.recursive(512,25);
        List<TextSegment> segments = documentSplitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        var embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        // Phase 2 RAG
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        // Boucle de dialogue avec l'assistant. L'utilisateur pose une question,
        // l'assistant répond en utilisant les informations du document.
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }


    }
}
