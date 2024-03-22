import 'dotenv/config';

import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import {ChatPromptTemplate, MessagesPlaceholder} from "@langchain/core/prompts";
import {StringOutputParser} from "@langchain/core/output_parsers";
import {HumanMessage, AIMessage} from "@langchain/core/messages";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {createRetrievalChain} from "langchain/chains/retrieval";
import {CheerioWebBaseLoader} from "langchain/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {createHistoryAwareRetriever} from "langchain/chains/history_aware_retriever";
import {createRetrieverTool} from "langchain/tools/retriever";
import {TavilySearchResults} from "@langchain/community/tools/tavily_search";
import {pull} from "langchain/hub";
import {createOpenAIFunctionsAgent, AgentExecutor} from "langchain/agents";

const chatModel = new ChatOpenAI({openAIApiKey: process.env.OPENAI_API_KEY});
const searchTool = new TavilySearchResults();
const splitter = new RecursiveCharacterTextSplitter();
const loader = new CheerioWebBaseLoader("https://docs.smith.langchain.com/user_guide");
const docs = await loader.load();
const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();
const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs,embeddings);

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
	["system","Answer the user's questions based on the below context:\n\n{context}"],
	new MessagesPlaceholder("chat_history"),
	["user", "{input}"],
]);
const historyAwareCombineDocsChain = await createStuffDocumentsChain({llm:chatModel, prompt:historyAwareRetrievalPrompt});

const retriever = vectorstore.asRetriever();
const conversationalRetrievalChain = await createRetrievalChain({combineDocsChain:historyAwareCombineDocsChain, retriever});

/*
const res = await conversationalRetrievalChain.invoke({chat_history: [
	new HumanMessage("Can LangSmith help test my LLM applications?"),
	new AIMessage("Yes!"),
],
	input: "tell me how"});
*/

// agent

const retrieverTool = await createRetrieverTool(retriever, {name: "langsmith_search", description: "Search for information about LangSmith. For any questions about LangSmith you must use this tool!"});
const tools = [retrieverTool, searchTool];

const agentPrompt = await pull("hwchase17/openai-functions-agent");
const agentModel = new ChatOpenAI({modelName: "gpt-3.5-turbo-1106", temperature: 0});
const agent = await createOpenAIFunctionsAgent({llm: agentModel, tools, prompt: agentPrompt});
const agentExecutor = new AgentExecutor({agent, tools, verbose: true});

const agentResult = await agentExecutor.invoke({input: "how can LangSmith help with testing?"});
console.log(agentResult.output);

