# Escopo da publicação

## Incluído no Git

- código-fonte de `src/` e `scripts/`;
- testes offline;
- configuração sem segredos;
- documentação técnica e científica revisada;
- artigo e apresentação finais em PDF;
- avaliação pequena com hashes, ambiente e métricas;
- CI, política de segurança, citação e aviso de direitos.

## Mantido local e ignorado

- `.env` e quaisquer tokens;
- 558 MB de `storage/` (dados, caches, modelos e saídas);
- vídeo MP4 de 173 MB;
- PPTX de 80 MB com mídias incorporadas;
- DOCX de edição, relatórios antiplágio e scripts de geração de entregas;
- termo assinado, currículo e documentos de estágio;
- configurações locais de ferramentas em `.claude/`;
- protótipo `hydraia-pre-tcc` e biblioteca de terceiros vendorizada.

## Motivos

1. **Segurança:** credenciais e caches podem expor acesso a provedores.
2. **Privacidade:** documentos administrativos contêm dados pessoais e assinaturas.
3. **Licença/proveniência:** dados e bibliotecas de terceiros exigem tratamento próprio.
4. **Tamanho:** GitHub não é o local apropriado para grandes dados, pesos e vídeo.
5. **Clareza científica:** código final e evidências auditadas não devem ser confundidos com experimentos exploratórios.

Se os artefatos pesados forem publicados depois, use um release, repositório de dados ou arquivo de pesquisa com DOI, acompanhado de hashes, licença e manifesto de proveniência. Git LFS sozinho resolve transporte, não resolve direitos nem reprodutibilidade.
