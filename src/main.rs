use rust_bert::{
    bart::{BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources},
    pipelines::{
        common::ModelType,
        question_answering::{QaInput, QuestionAnsweringConfig, QuestionAnsweringModel},
        sentence_embeddings::{
            SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
        },
    },
    resources::RemoteResource,
};
use std::{env::args, fs::read_to_string};

fn main() {
    let book = read_to_string(args().nth(1).unwrap()).unwrap();

    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllDistilrobertaV1)
        .create_model()
        .unwrap();

    let re = regex::Regex::new(r#"(.*?[\.?!]|["“”].*?["“”])"#).unwrap();

    let sentences = book
        .lines()
        .flat_map(|l| re.find(l).map(|m| m.as_str()))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();

    let out = model.encode(&sentences).unwrap();

    // println!("{:#?}", out);

    // let qa_model = QuestionAnsweringModel::new(QuestionAnsweringConfig {
    //     model_type: ModelType::Bart,
    //     model_resource: Box::new(RemoteResource::from_pretrained(
    //         BartModelResources::BART_CNN,
    //     )),
    //     config_resource: Box::new(RemoteResource::from_pretrained(
    //         BartConfigResources::BART_CNN,
    //     )),
    //     vocab_resource: Box::new(RemoteResource::from_pretrained(
    //         BartVocabResources::BART_CNN,
    //     )),
    //     merges_resource: Some(Box::new(RemoteResource::from_pretrained(
    //         BartMergesResources::BART_CNN,
    //     ))),

    //     ..Default::default()
    // })
    // .unwrap();

    // loop {
    //     let mut question = String::new();
    //     std::io::stdin().read_line(&mut question).unwrap();
    //     if question.is_empty() {
    //         break;
    //     }
    //     let answers = qa_model.predict(
    //         &[QaInput {
    //             question,
    //             context: context.clone(),
    //         }],
    //         1,
    //         32,
    //     );
    //     println!("{:?}", answers);
    // }
}
