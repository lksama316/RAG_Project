## 生成问题集

请帮根据eval\hak180产品安全手册_new.md，生成一组问题集（含答案）。
【格式】
1 要求 csv 格式
2 列头两个：question,answer
3 编码格式：UTF-8 BOM
【其他要求】
1 10道题
2 请以“HAK180 烫金机：”作为问题的开头
【保存位置】
保存位置在eval目录下，文件名为 qa.csv

## 生成评估程序

请帮我生成一个ragas的评估程序。

【输入】
1 读取评估程序当前目录下的qa.csv文件，其中question 列是问题，ground_truth 列是答案

【工具】
1 ragas评估程序中用到的llm 和 embedding，请通过调用以下模块获取：

- LLM：utils/llm_utils.py 中的 get_llm_client() 函数
- Embedding：utils/embedding_utils.py 中的 get_bge_m3_ef() 函数
  2 rag流程入口参考 processor/query_processor/main_graph.py 的 __main__：
- 使用 KBQueryWorkflow 类（含 run 方法，支持 stream=False 的 invoke 模式）
- 初始状态字段：original_query、session_id、is_stream
- 返回的 state 中通过 answer 取答案、reranked_docs 取上下文

【评估指标】
5个ragas指标: Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness

【输出】
1 输出格式为csv，保存在评估程序当前目录下qa_result.csv 文件中
2 输出9列: question、answer、context、ground_truth、faithfulness、answer_relevancy、context_precision、context_recall、answer_correctness
3 csv的文件编码为utf-8 BOM

【代码要求】
1 要求每个函数有对应的注释说明
2 核心步骤的函数用 step_1_xx step_2_xx step_3_xx ...为函数名的前缀
3 核心ragas函数 要有参数注释说明
