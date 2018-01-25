from utils.conlleval import return_report
import os
from utils.data_utils import Batch

def eval_ner(results, path, name):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, name + "_ner_predict.utf8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    f1 = float(eval_lines[1].strip().split()[-1])
    return eval_lines, f1

def evaluate(model, name, sentences, word_vocab, tag_vocab):
    batch_manager = Batch(sentences)
    results = []
    for batch in batch_manager.next_batch(shuffle=False):
        sources,lengths,predicts,golds = model.evaluate(*zip(*batch))
        for i in range(len(sources)):
            result = []
            sentence = [word_vocab[c] for c in sources[i][:lengths[i]]]
            predict = [tag_vocab[c] for c in predicts[i][:lengths[i]]]
            gold = [tag_vocab[c] for c in golds[i][:lengths[i]]]
            for c,g,p in zip(sentence,gold,predict):
                result.append(" ".join([c,g,p]))
            results.append(result)
    eval_lines, f1 = eval_ner(results, model.config.checkpoint_dir, name)
    for line in eval_lines:
        print(line)
    return f1


