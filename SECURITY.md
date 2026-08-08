# Política de segurança

## Dados que nunca devem ser publicados

- arquivos `.env` e chaves de API;
- caches de respostas de provedores;
- relatórios com prompts ou respostas que contenham tokens;
- documentos administrativos, currículos e termos assinados;
- pesos e datasets sem revisão de licença e proveniência.

Use apenas `.env.example` como referência. Se uma chave aparecer em commit, log, issue ou artefato, revogue-a no provedor antes de reescrever o histórico.

## Relato responsável

Não abra uma issue pública contendo credenciais ou dados pessoais. Entre em contato de forma privada com o autor do repositório e inclua apenas os passos mínimos para reprodução.

## Limite operacional

HydraIA é um protótipo acadêmico. Falhas de disponibilidade, calibração ou interpretação não devem ser tratadas como aconselhamento operacional para abastecimento público.
