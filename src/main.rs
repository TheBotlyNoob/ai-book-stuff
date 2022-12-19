use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};

static BEE_MOVIE: &str = include_str!("../bee-movie.txt");

fn main() {
    let qa_model = QuestionAnsweringModel::new(Default::default()).unwrap();

    let context = String::from(BEE_MOVIE);

    loop {
        let mut question = String::new();
        std::io::stdin().read_line(&mut question).unwrap();
        if question.is_empty() {
            break;
        }
        let answers = qa_model.predict(
            &[QaInput {
                question,
                context: context.clone(),
            }],
            1,
            32,
        );
        println!("{:?}", answers);
    }
}
