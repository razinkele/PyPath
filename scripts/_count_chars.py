from pathlib import Path
s = Path('scripts/extract_rpath_data.R').read_text()
print('{', s.count('{'), '}', s.count('}'))
print('(', s.count('('), ')', s.count(')'))
print("Quotes: ',", s.count("'"), ' ", '"', s.count('"'))
