num_layers=2
lr=0.001
epochs=20
dropout=0.5
hidden_size=64
seq_length=30
test_ratio=0.2

for seed in 16 32 64 128 256 512 1024 2048 4096 8192; do
    for model in "finbert" "roberta" "deberta"; do
        for target_column in "Binary_Price" "Float_Price" "Factor_Price" "Delta_Price"; do

            # Set feature columns based on sentiment model
            case $model in
                "deberta")
                    feature_columns="Close Volume total_news_count deberta_majority_vote deberta_count_positive deberta_count_negative deberta_count_neutral deberta_label_positive_sum deberta_label_negative_sum deberta_label_neutral_sum"
                    ;;
                "finbert")
                    feature_columns="Close Volume total_news_count finbert_majority_vote finbert_count_positive finbert_count_negative finbert_count_neutral finbert_label_positive_sum finbert_label_negative_sum finbert_label_neutral_sum"
                    ;;
                "roberta")
                    feature_columns="Close Volume total_news_count roberta_majority_vote roberta_count_positive roberta_count_negative roberta_count_neutral roberta_label_positive_sum roberta_label_negative_sum roberta_label_neutral_sum"
                    ;;
            esac

            for ticker in AAPL AMZN GOOGL META MSFT NVDA TSLA; do
                python stock_prediction/modeling/train.py \
                    --model_path models/lstm_price_${target_column}_${model}_model_${seed}.pth \
                    --data_path data/processed/${ticker}_preprocessed_dataset_with_features.csv \
                    --feature_columns $feature_columns \
                    --target_column $target_column \
                    --scaler_path models/${ticker}_lstm_price_${target_column}_${model}_scaler.pkl \
                    --seq_length $seq_length \
                    --hidden_size $hidden_size \
                    --num_layers $num_layers \
                    --dropout $dropout \
                    --test_ratio $test_ratio \
                    --lr $lr \
                    --epochs $epochs \
                    --force_retrain \
                    --seed $seed

                python stock_prediction/modeling/predict.py \
                    --model_path models/lstm_price_${target_column}_${model}_model_${seed}.pth \
                    --scaler_path models/${ticker}_lstm_price_${target_column}_${model}_scaler.pkl \
                    --data_path data/processed/${ticker}_preprocessed_dataset_with_features.csv \
                    --feature_columns $feature_columns \
                    --target_column $target_column \
                    --ticker ${ticker} \
                    --seq_length $seq_length \
                    --hidden_size $hidden_size \
                    --num_layers $num_layers \
                    --dropout $dropout \
                    --test_ratio $test_ratio \
                    --news_model $model \
                    --seed $seed
            done
        done
    done
done
