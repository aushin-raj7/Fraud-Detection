# import shap
# import pandas as pd
# import numpy as np

# def explain_model_predictions(model, X_data, num_cols, cat_cols, preproc, top_n=3, threshold=0.5):
#     """
#     Generates record-wise SHAP explanations for a trained model.
    
#     Parameters:
#         model : trained model (e.g., XGBClassifier)
#         X_data : preprocessed input data (numpy array or DataFrame)
#         num_cols : list of numerical feature names
#         cat_cols : list of categorical feature names
#         preproc : ColumnTransformer used for preprocessing
#         top_n : int, number of top features to include per record
#         threshold : float, decision threshold for binary classification
        
#     Returns:
#         recordwise_reasoning_df : pd.DataFrame with prediction, reasoning, and feature SHAP values
#     """
    
#     #Create SHAP explainer and values
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer(X_data)
    
#     #Combine numeric + one-hot encoded feature names
#     feature_names = []
#     feature_names.extend(num_cols)
#     cat_feature_names = preproc.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
#     feature_names.extend(cat_feature_names)
    
#     #Convert SHAP values to DataFrame
#     shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
#     shap_df['base_value'] = shap_values.base_values
#     shap_df['prediction_value'] = shap_values.base_values + shap_df.sum(axis=1)
    
#     #Predicted class based on threshold
#     pred_classes = (shap_df['prediction_value'] > threshold).astype(int)
    
#     #Function to get top features with reasoning
#     def get_top_features_with_reason(shap_row, n=top_n):
#         temp_df = pd.DataFrame({
#             'feature': feature_names,
#             'shap_value': shap_row
#         })
#         temp_df['abs_shap_value'] = temp_df['shap_value'].abs()
        
#         # Merge one-hot encoded columns back to original feature
#         temp_df['orig_feature'] = temp_df['feature'].str.split("_").str[0]
        
#         # Aggregate SHAP values by original feature
#         agg_df = temp_df.groupby('orig_feature').agg({'shap_value': 'sum'}).reset_index()
        
#         # Direction of contribution
#         agg_df['reason'] = agg_df['shap_value'].apply(lambda x: "increases" if x > 0 else "decreases")
#         agg_df['abs_shap_value'] = agg_df['shap_value'].abs()
        
#         # Top N features by absolute SHAP value
#         top_feats = agg_df.nlargest(n, 'abs_shap_value')
#         return top_feats[['orig_feature', 'shap_value', 'reason']]
    
#     #Build record-wise explanation
#     records_list = []
#     for idx, row in shap_df.iterrows():
#         pred_class = pred_classes[idx]
#         top_feats = get_top_features_with_reason(row[feature_names].values, n=top_n)
        
#         reasoning_text = "; ".join([f"{r['orig_feature']} {r['reason']}" for _, r in top_feats.iterrows()])
#         # shap_values_text = "; ".join([f"{r['orig_feature']} {r['shap_value']:.2f}" for _, r in top_feats.iterrows()])
#         # Convert top features to dictionary {feature: shap_value}
#         shap_values_dict = {r['orig_feature']: round(r['shap_value'], 2) for _, r in top_feats.iterrows()}
        
#         records_list.append({
#             'record_idx': idx,
#             'predicted_class': "Altered" if pred_class==1 else "Genuine",
#             'base_value': row['base_value'],
#             'prediction_value': row['prediction_value'],
#             'reasoning': reasoning_text,
#             'feature_shap_values': shap_values_dict
#         })
    
#     recordwise_reasoning_df = pd.DataFrame(records_list)
#     return recordwise_reasoning_df



import shap
import pandas as pd
import numpy as np
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

def explain_model_predictions(model, X_data, num_cols, cat_cols, preproc, top_n=3, threshold=0.5):
    """
    Generates record-wise SHAP explanations for a trained model.
    """
    
    try:
        logger.info("Creating SHAP explainer and computing SHAP values")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_data)
        
        # ROBUST METHOD: Get feature names using get_feature_names_out()
        num_features = num_cols
        
        # Get the OneHotEncoder from the preprocessor
        ohe_encoder = preproc.named_transformers_['cat']['ohe']
        
        # Get one-hot encoded feature names - this works across scikit-learn versions
        cat_features = ohe_encoder.get_feature_names_out(cat_cols)
        
        # Combine all feature names
        all_features = np.concatenate([num_features, cat_features])
        
        logger.info(f"Feature composition - Numerical: {len(num_features)}, "
                   f"One-hot encoded: {len(cat_features)}, Total: {len(all_features)}")
        
        # Convert SHAP values to DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=all_features)
        shap_df['base_value'] = shap_values.base_values
        shap_df['prediction_value'] = shap_values.base_values + shap_df.sum(axis=1)
        
        # Predicted class based on threshold
        pred_classes = (shap_df['prediction_value'] > threshold).astype(int)
        
        # Function to get top features with reasoning
        def get_top_features_with_reason(shap_row, n=top_n):
            temp_df = pd.DataFrame({
                'feature': all_features,
                'shap_value': shap_row
            })
            temp_df['abs_shap_value'] = temp_df['shap_value'].abs()
            
            # Map features back to original names (for one-hot encoded features)
            def get_original_feature(feature_name):
                if feature_name in num_features:
                    return feature_name
                else:
                    # For one-hot encoded features, extract original feature name
                    # Handle cases where original feature name might contain underscores
                    for cat_col in cat_cols:
                        if feature_name.startswith(cat_col + '_'):
                            return cat_col
                    # If no direct match, try to extract by splitting on last underscore
                    parts = feature_name.rsplit('_', 1)
                    return parts[0] if len(parts) > 1 else feature_name
            
            temp_df['orig_feature'] = temp_df['feature'].apply(get_original_feature)
            
            # Aggregate SHAP values by original feature
            agg_df = temp_df.groupby('orig_feature').agg({'shap_value': 'sum'}).reset_index()
            
            # Direction of contribution
            agg_df['reason'] = agg_df['shap_value'].apply(lambda x: "increases" if x > 0 else "decreases")
            agg_df['abs_shap_value'] = agg_df['shap_value'].abs()
            
            # Top N features by absolute SHAP value
            top_feats = agg_df.nlargest(n, 'abs_shap_value')
            return top_feats[['orig_feature', 'shap_value', 'reason']]
        
        # Build record-wise explanation
        logger.info(f"Building explanations for {len(shap_df)} records")
        records_list = []
        for idx, row in shap_df.iterrows():
            pred_class = pred_classes[idx]
            top_feats = get_top_features_with_reason(row[all_features].values, n=top_n)
            
            reasoning_text = "; ".join([f"{r['orig_feature']} {r['reason']}" for _, r in top_feats.iterrows()])
            
            # Convert top features to dictionary {feature: shap_value}
            shap_values_dict = {r['orig_feature']: round(r['shap_value'], 2) for _, r in top_feats.iterrows()}
            
            records_list.append({
                'record_idx': idx,
                'predicted_class': "Altered" if pred_class == 1 else "Genuine",
                'base_value': round(row['base_value'], 4),
                'prediction_value': round(row['prediction_value'], 4),
                'reasoning': reasoning_text,
                'feature_shap_values': shap_values_dict
            })
        
        recordwise_reasoning_df = pd.DataFrame(records_list)
        logger.info("Successfully computed SHAP explanations")
        return recordwise_reasoning_df
        
    except Exception as e:
        logger.exception("Error in SHAP explanation computation")
        raise