import pandas as pd
import numpy as np

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu, norm, chi2

from sklearn.metrics import accuracy_score, f1_score, classification_report,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def compare_datasets(df1, df2, parameter_types, alpha=0.05, correction='bonferroni'):
    """
    Compares two datasets for similarity across parameters.
    
    Parameters:
        df1, df2: pandas DataFrame
            The two datasets to compare.
        parameter_types: dict
            Dictionary mapping parameter names to their types: 'continuous', 'nominal', or 'ordinal'.
        alpha: float
            Significance level for hypothesis testing.
        correction: str
            Multiple comparison correction method ('bonferroni', 'fdr_bh', etc.).
    
    Returns:
        results: pd.DataFrame
            Summary of test results for each parameter.
    """
    results = []
    
    for param, param_type in parameter_types.items():
        
        # Skip if parameter not in both datasets
        if param not in df1.columns or param not in df2.columns:
            continue
        
        # Continuous variables
        if param_type == 'continuous':
            stat, p_val = ks_2samp(df1[param].dropna(), df2[param].dropna())
            test_name = "Kolmogorov-Smirnov"
        
        # Nominal categorical variables
        elif param_type == 'nominal':
            contingency_table = pd.crosstab(df1[param].dropna(), df2[param].dropna())
            stat, p_val, _, _ = chi2_contingency(contingency_table)
            test_name = "Chi-Square"
        
        # Ordinal categorical variables
        elif param_type == 'ordinal':
            stat, p_val = mannwhitneyu(df1[param].dropna(), df2[param].dropna(), alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        else:
            continue
        
        results.append({'Parameter': param, 'Type': param_type, 'Test': test_name, 'Statistic': stat, 'p-value': p_val})
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    if correction:
        corrected_pvals = multipletests(results_df['p-value'], alpha=alpha, method=correction)
        results_df['Corrected p-value'] = corrected_pvals[1]
        results_df['Significant'] = corrected_pvals[0]
    else:
        results_df['Significant'] = results_df['p-value'] < alpha
    
    return results_df

def apply_VIF(df, columns_to_drop_vif = []):
    '''
    Calculate the Variance Inflation Factor (VIF) for each feature in the dataset.
    
    Parameters:
        df: pd.DataFrame
            The dataset to calculate VIF for.
        columns_to_drop_vif: list
            List of columns to drop before calculating VIF.
            
    Returns:
        vif_data: pd.DataFrame
            DataFrame containing the VIF values for each feature.
    '''
    
    # Initialize the DataFrame to store the VIF values
    vif_data = pd.DataFrame()
    X_to_test = df.copy()

    # Drop the columns with high correlation
    if len(columns_to_drop_vif) > 0:
        X_to_test = X_to_test.drop(columns = columns_to_drop_vif)
    
    # Add a constant to the features
    X_to_test = sm.add_constant(X_to_test)
    
    # Calculate the VIF for each feature
    vif_data["feature"] = X_to_test.columns
    vif_data["VIF"] = [variance_inflation_factor(X_to_test.values, i) for i in range(X_to_test.shape[1])]
    
    return vif_data

def display_coefficients(X_train, Y_train,multi_output_model):
    '''
    Display the coefficients of the logistic regression model for each target variable
    
    Args:
        X_train (DataFrame): The independent variables of the training set
        Y_train (DataFrame): The dependent variables of the training set
        multi_output_model (MultiOutputClassifier): The fitted multi-output model
    
    Returns:
    Data    Frame: The coefficients of the logistic regression model for each target
    '''
    # Create a dictionary to store the coefficients for each target variable
    coef_dict = {}
    # Iterate over the target variables
    for i in range(len(multi_output_model.estimators_)):
        # Get the coefficients for each target variable
        coefficients = np.concatenate([multi_output_model.estimators_[i].intercept_.reshape(-1),multi_output_model.estimators_[i].coef_.reshape(-1)])
        coef_dict[Y_train.columns[i]] = coefficients
    # Create a DataFrame from the dictionary
    coefficients_df = pd.DataFrame(coef_dict).T
    # Add the column names
    coefficients_df.columns = ['intercept'] + list(X_train.columns)
    return coefficients_df

def calculate_score_metrics(Y_true, Y_pred , verbose = False):
    '''
    Calculate the accuracy and F1 score for each dimension of the multi-output classification problem.
    
    Parameters:
        Y_true: pd.DataFrame
            The true labels.
        Y_pred: np.array
            The predicted labels.
        verbose: bool
            If True, print the results for each dimension.
            
    Returns:
        results: pd.DataFrame
            DataFrame containing the accuracy and F1 score for each dimension.
    '''

    # Initialize the results dictionary
    results = {}
    
    # Calculate the accuracy and F1 score for each dimension
    for y_dim in range(Y_true.shape[1]):
        accuracy = accuracy_score(Y_true.values[:, y_dim], Y_pred[:, y_dim]) # Calculate accuracy
        f1 = f1_score(Y_true.values[:, y_dim], Y_pred[:, y_dim]) # Calculate F1 score
        class_report = classification_report(Y_true.values[:, y_dim], Y_pred[:, y_dim]) # Get the classification report
        
        # Print the results if verbose is True
        if verbose:
            print(f"Dimension {Y_true.columns[y_dim]}")
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print('Classification Report:\n', class_report, '\n\n')
        
        # Store the results in the dictionary
        results[Y_true.columns[y_dim]] = [accuracy, f1]
        
    # Create a DataFrame from the results dictionary
    results = pd.DataFrame(results, index=['Accuracy', 'F1 Score']).T
    
    return results

def logit_pvalue(model, x):
    '''
    Calculate the p-values for the coefficients of a logistic regression model.
    
    Parameters:
        model: LogisticRegression
            The logistic regression model.
        x: np.array
            The input features.
    
    Returns:
        p: np.array
            The p-values for the coefficients.
    '''
    p = model.predict_proba(x) # get probabilities
    n = len(p) # number of samples
    m = len(model.coef_[0]) + 1 # number of coefficients - adding one for intercept
    coefs = np.concatenate([model.intercept_, model.coef_[0]]) # coefficients
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1)) # insert 1's for intercept
    
    ans = np.zeros((m, m)) # prepare Hessian
    
    # calculate Hessian
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    
    # Invert Hessian
    ans = np.array(ans, dtype=float)
    vcov = np.linalg.inv(ans)
    
    se = np.sqrt(np.diag(vcov)) # Standard errors
    t =  coefs/se # t scores
    p = (1 - norm.cdf(abs(t))) * 2 # p values
    
    return p

def calculate_p_value_variable(X, Y, model_estimators):
    '''
    Calculate the p-values for the coefficients of a logistic regression model for each dimension.
    
    Parameters:
        X: pd.DataFrame
            The input features.
        Y: pd.DataFrame
            The target variables.
        model_estimators: list
            List of logistic regression models for each dimension.
    
    Returns:
        df_significance: pd.DataFrame
            DataFrame containing the p-values for each coefficient
    '''
    
    # Initialize the DataFrame to store the p-values
    col_names_sig = ['intercept'] + X.columns.to_list()
    df_significance = pd.DataFrame(columns=col_names_sig, index = Y.columns)
    
    # Calculate the p-values for each dimension
    for i in range(len(model_estimators)):
        # Calculate the p-values for the coefficients
        df_significance.loc[Y.columns[i]] = logit_pvalue(model_estimators[i], X)
        
        # Set the p-values to NaN for the coefficients that are zero
        idx = model_estimators[i].coef_ == 0
        idx_proper = [False] + idx[0].tolist()
        idx_cols = df_significance.columns[idx_proper]
        if len(idx_cols) > 0:
            df_significance.loc[Y.columns[i], idx_cols] = np.nan
            
    return df_significance

def calculate_p_value_model(X, Y, estimators):
    '''
    Calculate the p-values for the logistic regression model for each dimension.
    
    Parameters:
        X: pd.DataFrame
            The input features.
        Y: pd.DataFrame
            The target variables.
        estimators: list
            List of logistic regression models for each dimension.
    
    Returns:
        df_results: pd.DataFrame
            DataFrame containing the likelihood ratio statistic and p-value for each model.
    '''
    
    
    # Fit the null model (intercept-only model)
    null_model = MultiOutputClassifier(LogisticRegression(fit_intercept=True, max_iter=1000), n_jobs=-1)
    null_model.fit(np.zeros_like(X), Y)  # Null model (predicting all zeros)

    # Initialize a dictionary to store the results
    store_values = {}
    
    # For each estimator:
    for estimator_idx in range(len(estimators)):
        
        # Get the current estimator, null model, and target variable
        current_estimator = estimators[estimator_idx]
        current_null = null_model.estimators_[estimator_idx]
        current_target = Y.values[:, estimator_idx]

        # Calculate the log-likelihood for the full and null models
        log_likelihood_full = -log_loss(current_target, current_estimator.predict_proba(X))
        log_likelihood_null = -log_loss(current_target, current_null.predict_proba(X))

        # Calculate the likelihood ratio statistic
        lr_statistic = 2 * (log_likelihood_full - log_likelihood_null)

        # Degrees of freedom is the number of predictors in the model
        df = X.shape[1]  # Number of features

        # Compute the p-value for the likelihood ratio test
        p_value = 1 - chi2.cdf(lr_statistic, df)

        # Store the results in the dictionary
        store_values[Y.columns[estimator_idx]] = [round(lr_statistic,4), round(p_value,4)]

    # Create a DataFrame to store the results
    df_results = pd.DataFrame(store_values, index=['Likelihood Ratio Statistic', 'p-value']).T
    return df_results