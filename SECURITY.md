# Política de segurança

> HydraIA is a
> [Brazilian Undergraduate Thesis (TCC)](docs/ACADEMIC_CONTEXT.md) and final-year
> Computer Engineering capstone project, not a production water-management
> service.

## Dados que nunca devem ser publicados

- arquivos `.env` e chaves de API;
- caches de respostas de provedores;
- relatórios com prompts ou respostas que contenham tokens;
- documentos administrativos, currículos e termos assinados;
- pesos e datasets sem revisão de licença e proveniência.

Use apenas `.env.example` como referência. Se uma chave aparecer em commit, log, issue ou artefato, revogue-a no provedor antes de reescrever o histórico.

## Componentes, credenciais e fluxo

HydraIA é um processo CLI/batch executado por um usuário do sistema operacional.
Ele não emite sessões, JWTs ou refresh tokens e não oferece um servidor de login.
A autorização para ler arquivos, executar modelos e acessar o SQLite depende das
permissões locais. As chaves abaixo autenticam **chamadas de saída** aos provedores.

| Componente | Credencial e uso |
|---|---|
| `EnvLoader` | Lê o `.env` da raiz do projeto durante a importação e preenche somente variáveis ainda ausentes no ambiente. O formato local é `CHAVE=valor`, sem expansão de variáveis. |
| OpenWeather | `OPENWEATHER_API_KEY` é enviada como parâmetro `appid` por HTTPS, tanto na sondagem geográfica quanto na coleta. A chave ainda faz parte da URL recebida pelo provedor. |
| Especialistas Gemini e GPT | `GEMINI_ULTRA_API_KEY` e `OPENAI_GPT5_API_KEY` são passadas aos respectivos SDKs. Os prompts necessários à análise são enviados aos provedores externos. |
| LM Studio | Não recebe essas chaves. Descobre o modelo e envia o prompt de síntese ao endereço configurado. Não há autenticação de usuário implementada por este cliente. |
| Cache e persistência | Respostas, relatórios e metadados podem ser gravados em JSON, Markdown e SQLite dentro de `storage/`; esses arquivos não são criptografados pela aplicação. |

O CLI carrega a configuração, verifica a presença/formato básico das chaves e,
quando solicitado, sonda os provedores. A validação local de formato não comprova
que uma chave está autorizada. O coletor meteorológico envia a chave separadamente
em `params`; os especialistas enviam seus prompts via SDK. Após a validação dos
resultados, o sintetizador recebe o contexto e a saída é persistida localmente.

As chaves permanecem no ambiente e na memória do processo. Não as inclua em
prompts, argumentos de linha de comando, URLs de configuração ou conteúdo de
relatórios. Proteja `.env` e `storage/` com permissões do sistema operacional.
Para rotacionar uma chave, revogue/substitua no provedor, atualize a origem efetiva
da variável e reinicie os processos; um valor já exportado tem precedência sobre
o `.env`.

## Transporte e diagnóstico

O coletor OpenWeather exige uma URL HTTPS sem credenciais embutidas, query ou
fragmento. Sondagens de OpenWeather e listagens auxiliares de modelos usam HTTPS.
Esses clientes não seguem redirecionamentos. A chave de geocoding continua em
`appid`, conforme o contrato do provedor; HTTPS não impede que o provedor ou um
proxy TLS autorizado tenha acesso a essa URL.

Para LM Studio, HTTP só é aceito com `localhost` ou um endereço IP de loopback.
Um destino remoto exige HTTPS, inclusive em `LM_STUDIO_NATIVE_CHAT_URL`. Descoberta,
chat nativo, retry de compatibilidade e fallback legado recusam redirecionamentos,
evitando reenviar prompts para um destino indicado na resposta. Mantenha o
servidor local vinculado ao loopback. HTTPS protege o transporte; se publicar um
servidor remoto, o controle de acesso deve ser configurado nesse servidor/gateway.
Um servidor que exija token no cliente precisa de uma integração adicional: não
coloque esse token na URL para contornar a ausência de suporte.

A verificação de certificados permanece ativa. O bootstrap pode usar as CAs
nativas via `truststore`; `HYDRAIA_NATIVE_CA=0` desativa apenas essa integração,
não a validação TLS. Endereços de provedores, proxies e demais variáveis do
ambiente são configuração confiável do operador. Este cliente não é uma sandbox
contra um operador que controle o processo ou seu ambiente.

Diagnósticos de HTTP e de configuração evitam imprimir URLs com chaves, corpos de
erro ou valores inválidos. Os handlers instalados por `setup_logging` mascaram
padrões conhecidos de credenciais nas mensagens. Exceções e stacks preservam
classes e frames, mas omitem mensagens, notas, linhas de código e variáveis locais
que poderiam repetir segredos. Esse mascaramento é uma defesa adicional e não
uma garantia de reconhecer qualquer segredo em texto arbitrário. Handlers
externos, dumps de memória e saídas que o usuário adicionar exigem cuidado próprio.

As correções de transporte e diagnóstico são posteriores à entrega acadêmica;
não reavaliam nem alteram as métricas, os datasets, os checkpoints ou o artigo.

## Relato responsável

Não abra uma issue pública contendo credenciais ou dados pessoais. Entre em contato de forma privada com o autor do repositório e inclua apenas os passos mínimos para reprodução.

## Limite operacional

HydraIA é um protótipo acadêmico. Falhas de disponibilidade, calibração ou interpretação não devem ser tratadas como aconselhamento operacional para abastecimento público.
