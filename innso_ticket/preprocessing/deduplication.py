import re

import pandas as pd

import config

# customer service template
CU_TEMPLATE = {
    "english":
        ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
    ,
    "german":
        ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
         "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
    ,
    "french":
        ["L'équipe d'assistance à la clientèle d'Aspiegel\,?",
         "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
    ,
    "spanish":
        ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
         "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
    ,
    "italian":
        ["Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
         "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
    ,
    "portuguese":
        ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
         "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
         "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
    ,
}

CU_PATTERN = '|'.join((map(lambda x: f"({x})", sum(list(CU_TEMPLATE.values()), []))))

# email split template
SPLIT_TEMPLATES = [
    "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)",
    "(On.{30,60}wrote:)",
    "(Re\s?:|RE\s?:)",
    "(\*\*\*\*\*\(PERSON\) Support issue submit)",
    "(\s?\*\*\*\*\*\(PHONE\))*$",
]

SPLIT_PATTERN = '|'.join(SPLIT_TEMPLATES)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([config.TICKET_ID_COL, config.INTERACTION_ID_COL]).reset_index(drop=True)

    # start processing ticket data
    for ticket, sdf in df.groupby(config.TICKET_ID_COL):
        # for one ticket content data
        text_set = set([])
        deduplicated_texts = []
        for text in sdf[config.CONTENT_COL]:

            texts = re.split(SPLIT_PATTERN, text)

            texts = [x for x in texts if x is not None]

            # replace split patterns
            texts = [re.sub(SPLIT_PATTERN, '', x.strip()) for x in texts]

            # replace customer template
            texts = [re.sub(CU_PATTERN, '', x.strip()) for x in texts]

            texts = [x for x in texts if len(x) > 0]

            deduplicated = []
            for x in texts:
                if x not in text_set:
                    text_set.add(x)
                    deduplicated.append(x)

            deduplicated_texts.append('\n'.join(deduplicated))

        df.loc[df[config.TICKET_ID_COL] == ticket, config.CONTENT_COL] = deduplicated_texts
    return df
