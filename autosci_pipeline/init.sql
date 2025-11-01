# init.sql - Initialisation PostgreSQL
CREATE DATABASE autosci;
CREATE DATABASE mlflow;

-- Créer un utilisateur pour l'application
CREATE USER autosci_user WITH PASSWORD 'autosci_pass';
GRANT ALL PRIVILEGES ON DATABASE autosci TO autosci_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO autosci_user;

-- Tables pour la plateforme de déploiement
\c autosci;

CREATE TABLE IF NOT EXISTS deployed_models (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    framework VARCHAR(100),
    model_type VARCHAR(100),
    metrics TEXT,
    deployment_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    model_path VARCHAR(500),
    pricing TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255),
    purchase_type VARCHAR(50),
    price DECIMAL(10,2),
    email VARCHAR(255),
    transaction_id VARCHAR(255),
    status VARCHAR(50),
    created_at TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    model_id VARCHAR(255),
    subscription_type VARCHAR(50),
    status VARCHAR(50),
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);

CREATE INDEX idx_deployed_models_run_id ON deployed_models(run_id);
CREATE INDEX idx_transactions_model_id ON transactions(model_id);
CREATE INDEX idx_subscriptions_email ON subscriptions(email);