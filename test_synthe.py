from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
import os 
os.environ["OPENAI_API_KEY"]="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"
synthesizer = Synthesizer(model="gpt-4",multithreading=True)
contexts =[
           ["Lockdown was implemented on March, 2020 in India.Coronaviruses are a large family of respiratory viruses that includes COVID-19, Middle East Respiratory Syndrome (MERS), and Severe Acute Respiratory Syndrome (SARS)."],
           [ "Coronaviruses cause diseases in animals and humans. They often circulate among camels, cats, and bats, and can sometimes evolve and infect people."],
            ["Its symptoms depend on the virus, but in humans common signs include mild respiratory infections, like the common cold, fever, cough, shortness of breath, and breathing difficulties"],
            ["In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death."],
            ["There are two hypotheses as to COVID-19's origins: exposure to an infected animal or a laboratory leak. There is not enough evidence to support either argument."],
           [ "The novel coronavirus (SARS-CoV-2) that causes COVID-19 first emerged in the Chinese city of Wuhan in 2019 and was declared a pandemic by the World Health Organization (WHO)."],
           [ "The 2020 lockdown in India left tens of millions of migrant workers unemployed. With factories and workplaces shut down, many migrant workers were left with no livelihood. They thus decided to walk hundreds of kilometers to go back to their native villages, accompanied by their families in many cases. In response, the central and state governments took various measures to help them. The central government then announced that it had asked state governments to set up immediate relief camps for the migrant workers returning to their native states,and later issued orders protecting the rights of the migrants."]
    
 ]

# Use synthesizer directly
synthesizer.generate_goldens(contexts=contexts)
synthesizer.save_as(
    # also accepts 'csv'
    file_type='json',
    directory="./synthetic_data"
)


# Use synthesizer within an EvaluationDataset
dataset = EvaluationDataset()
dataset.generate_goldens(
    synthesizer=synthesizer,
    contexts=contexts
)
dataset.save_as(
    # also accepts 'csv'
    file_type='json',
    directory="./synthetic_data"
)  