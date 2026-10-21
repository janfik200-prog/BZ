# Проверки

Без базы, модели и видеокарты — то же, что в CI (.github/workflows/ci.yml):

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

Проверки, которым нужны docling или matplotlib, без них пропускаются сами. Большие
проверки разработчика и трассировка узких вопросов на живой базе — в `.local`
(`python georag.py test`, `.local/tools`), в git не идут.
