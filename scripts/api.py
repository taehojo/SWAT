import requests
from tqdm import tqdm

def get_snp_details(top_20_features):
    enriched_top_20_features = []

    for feature in tqdm(top_20_features, desc='Getting SNP details'):
        rs_id = feature[0]
        importance = feature[1]

        try:
            response = requests.get(f'https://clinicaltables.nlm.nih.gov/api/snps/v3/search?terms={rs_id}&q={rs_id}')
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error getting SNP details for {rs_id}: {e}")
            position, alleles, gene = ("Unknown", "Unknown", "Unknown")
        else:
            data = response.json()
            _, _, _, details = data
            if details and len(details[0]) >= 5:
                detail = details[0]
                rs_id, chromosome, position, alleles, gene = detail
                position = f"{chromosome}:{position}"
            else:
                position, alleles, gene = ("Unknown", "Unknown", "Unknown")

        enriched_top_20_features.append({
            'rs_id': rs_id,
            'importance': importance,
            'position': position,
            'alleles': alleles,
            'gene': gene
        })

    return enriched_top_20_features

