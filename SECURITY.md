# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Alpha Quarry, please report it responsibly.

### How to Report

1. **Do NOT open a public GitHub issue** for security vulnerabilities.

2. **Email**: Send a description of the vulnerability to pengfeijiang320@gmail.com or contact the maintainer directly via GitHub private message.

3. **Include the following information**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Assessment**: We will assess the vulnerability and determine its severity.
- **Fix**: We will work on a fix and release it as soon as possible.
- **Disclosure**: We will coordinate with you on the disclosure timeline.

## Security Best Practices

When using Alpha Quarry, please follow these security best practices:

### Credential Management

- **Never commit credentials to version control**
  - Use environment variables for sensitive configuration (e.g., `TUSHARE_TOKEN`)
  - Use `configs/datasource.local.yaml` for local overrides (already in `.gitignore`)
  - Copy `configs/datasource.example.yaml` as a template

- **Environment variable setup**:
  ```bash
  # Linux/macOS
  export TUSHARE_TOKEN="your_token_here"

  # Windows PowerShell
  $env:TUSHARE_TOKEN = "your_token_here"
  ```

### Data Security

- The `data/` directory contains market data and analysis results
- Do not share personal trading data or proprietary factors publicly
- Review `.gitignore` before committing to ensure sensitive files are excluded

### Dependencies

- Regularly update dependencies to patch known vulnerabilities
- Run `pip audit` or similar tools to check for vulnerable dependencies
- Review `requirements.txt` and `pyproject.toml` for dependency versions

## Security Features

- **Preflight Guard**: Run `python scripts/preflight_guard.py --strict` to check for common security issues before committing
- **Gitignore**: Comprehensive `.gitignore` to prevent accidental commits of sensitive files
- **No Hardcoded Credentials**: The project uses environment variables and configuration files for all credentials

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged in the project (with their permission).
