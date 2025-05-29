import pandas as pd
import glob 


def main():
    # Load all scored JSONL files
    files = glob.glob("*scored.jsonl")
    dfs = []
    for file in files:
        tmp = pd.read_json(file, lines=True, orient='records')
        dfs.append(tmp)
    df = pd.concat(dfs)   

    # 1) Compute per‐row overall score
    df['overall'] = (
          0.4 * df['correctness']
        + 0.2 * df['coherence']
        + 0.2 * df['detail']
        + 0.2 * df['fluency']
    )

    # 2) Split MCQ/TF vs. other
    mcq_tf_mask = df['question_type'].isin(['true_false', 'multiple_choice'])
    mcq_tf = df[mcq_tf_mask]
    others = df[~mcq_tf_mask]

    # 3a) Accuracy for MCQ & TF, per model
    accuracy_summary = (
        mcq_tf
        .groupby('model_name')['score']
        .mean()
        .reset_index(name='accuracy_mcq_tf')
    )

    # 3b) Averages of cas, correctness, coherence, detail, fluency, and overall for other questions
    others_summary = (
        others
        .groupby('model_name')
        .agg(
            cas_avg          = ('cas',       'mean'),
            correctness_avg  = ('correctness', 'mean'),
            coherence_avg    = ('coherence', 'mean'),
            detail_avg       = ('detail',    'mean'),
            fluency_avg      = ('fluency',   'mean'),
            overall_avg      = ('overall', 'mean'),
        )
        .reset_index()
    )

    # 4) Merge into one table
    summary = pd.merge(
        accuracy_summary,
        others_summary,
        on='model_name',
        how='outer'
    ).fillna(0)

    # 5) Write to csv
    summary.to_csv('summary.csv', index=False)
    print("Summary saved to summary.csv")


if __name__ == "__main__":
    main()