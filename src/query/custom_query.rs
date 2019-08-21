use crate::schema::IndexRecordOption;
use crate::query::{Explanation, Weight, Scorer, Query};
use crate::query::bm25::BM25Weight;
use std::collections::BTreeSet;
use crate::{Term, Searcher, SegmentReader, SkipResult, Postings, DocSet, DocId, Score};
use crate::query::explanation::does_not_match;
use crate::postings::SegmentPostings;
use crate::fieldnorm::FieldNormReader;
use crate::Result;
use std::fmt::Debug;

/*
#[derive(Debug, Clone)]
pub struct CustomTerm {
    pub text : String,
    pub field : u32
}
*/


pub trait CustomWeightFunction : Debug + Clone + 'static
{
    fn get_weight(&self,text : String, field: u32) -> f32;
}

/*impl CustomWeightFunction for CustomTerm {
    fn get_weight(&self) -> f32 {
        match (self.text.as_ref(),self.field) {
            ("DH", 0) => {32.0}
            ("ALL", 0) => {16.0}
            ("NON-CHRONO", 1) => {12.0}
            ("ALL", 1)  => {8.0}
            ("965", 2) => {4.0}
            ("ALL", 2) => {2.0}
            (_,_) => {0.0}
        }

    }
}*/

/*impl CustomTerm {

    fn get_weight(&self) -> f32 {
        match (self.text.as_ref(),self.field) {
            ("DH", 0) => {32.0}
            ("ALL", 0) => {16.0}
            ("NON-CHRONO", 1) => {12.0}
            ("ALL", 1)  => {8.0}
            ("965", 2) => {4.0}
            ("ALL", 2) => {2.0}
            (_,_) => {0.0}
        }

    }
}*/

#[derive(Clone, Debug)]
pub struct CustomTermQuery<T:CustomWeightFunction> {
    term: Term,
    index_record_option: IndexRecordOption,
    custom_weight : T
}

/*fn get_weight<T:CustomWeightFunction>(t : &T) ->f32 {
    t.get_weight()
}*/

impl <T:CustomWeightFunction> CustomTermQuery<T> {
    /// Creates a new term query.
    pub fn new(term: Term, segment_postings_options: IndexRecordOption,custom_weight:T ) ->CustomTermQuery<T> {
      CustomTermQuery {
            term,
            index_record_option: segment_postings_options,
            custom_weight
        }
    }

    /// The `Term` this query is built out of.
    pub fn term(&self) -> &Term {
        &self.term
    }




    /// Returns a weight object.
    ///
    /// While `.weight(...)` returns a boxed trait object,
    /// this method return a specific implementation.
    /// This is useful for optimization purpose.
    pub fn specialized_weight(&self, searcher: &Searcher, scoring_enabled: bool) -> CustomTermWeight {
        let term = self.term.clone();
        //let mut bm25_weight = BM25Weight::for_terms(searcher, &[term]);
        let bm25_weight = BM25Weight {
            idf_explain: Explanation::new("",0.0),
            //text:term.text().to_string(),field:term.field().0
            weight :self.custom_weight.get_weight(term.text().to_string(),term.field().0),
            //weight: CustomTerm {text:term.text().to_string(),field:term.field().0}.get_weight(),
            cache: [0f32; 256],
            average_fieldnorm: 0.0,
        };
        let index_record_option = if scoring_enabled {
            self.index_record_option
        } else {
            IndexRecordOption::Basic
        };
        CustomTermWeight::new(self.term.clone(), index_record_option, bm25_weight)
    }
}

impl <T:CustomWeightFunction> Query for CustomTermQuery<T> {
    fn weight(&self, searcher: &Searcher, scoring_enabled: bool) -> Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight(searcher, scoring_enabled)))
    }
    fn query_terms(&self, term_set: &mut BTreeSet<Term>) {
        term_set.insert(self.term.clone());
    }
}

pub struct CustomTermWeight {
    term: Term,
    index_record_option: IndexRecordOption,
    similarity_weight: BM25Weight,
}

impl Weight for CustomTermWeight {
    fn scorer(&self, reader: &SegmentReader) -> Result<Box<dyn Scorer>> {
        let term_scorer = self.scorer_specialized(reader)?;
        Ok(Box::new(term_scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> Result<Explanation> {
        let mut scorer = self.scorer_specialized(reader)?;
        if scorer.skip_next(doc) != SkipResult::Reached {
            return Err(does_not_match(doc));
        }
        Ok(scorer.explain())
    }

    fn count(&self, reader: &SegmentReader) -> Result<u32> {
        if let Some(delete_bitset) = reader.delete_bitset() {
            Ok(self.scorer(reader)?.count(delete_bitset))
        } else {
            let field = self.term.field();
            Ok(reader
                .inverted_index(field)
                .get_term_info(&self.term)
                .map(|term_info| term_info.doc_freq)
                .unwrap_or(0))
        }
    }
}

impl CustomTermWeight {
    pub fn new(
        term: Term,
        index_record_option: IndexRecordOption,
        similarity_weight: BM25Weight,
    ) -> CustomTermWeight {
        CustomTermWeight {
            term,
            index_record_option,
            similarity_weight,
        }
    }

    fn scorer_specialized(&self, reader: &SegmentReader) -> Result<CustomTermScorer> {
        let field = self.term.field();
        let inverted_index = reader.inverted_index(field);
        let fieldnorm_reader = reader.get_fieldnorms_reader(field);
        let similarity_weight = self.similarity_weight.clone();
        let postings_opt: Option<SegmentPostings> =
            inverted_index.read_postings(&self.term, self.index_record_option);
        if let Some(segment_postings) = postings_opt {
            Ok(CustomTermScorer::new(
                segment_postings,
                fieldnorm_reader,
                similarity_weight,
            ))
        } else {
            Ok(CustomTermScorer::new(
                SegmentPostings::empty(),
                fieldnorm_reader,
                similarity_weight,
            ))
        }
    }
}

pub struct CustomTermScorer {
    postings: SegmentPostings,
    fieldnorm_reader: FieldNormReader,
    similarity_weight: BM25Weight,
}

impl CustomTermScorer {
    pub fn new(
        postings: SegmentPostings,
        fieldnorm_reader: FieldNormReader,
        similarity_weight: BM25Weight,
    ) -> CustomTermScorer {
        CustomTermScorer {
            postings,
            fieldnorm_reader,
            similarity_weight,
        }
    }
}

impl CustomTermScorer {
    pub fn term_freq(&self) -> u32 {
        self.postings.term_freq()
    }

    pub fn fieldnorm_id(&self) -> u8 {
        self.fieldnorm_reader.fieldnorm_id(self.doc())
    }

    pub fn explain(&self) -> Explanation {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.explain(fieldnorm_id, term_freq)
    }
}

impl DocSet for CustomTermScorer {
    fn advance(&mut self) -> bool {
        self.postings.advance()
    }

    fn skip_next(&mut self, target: DocId) -> SkipResult {
        self.postings.skip_next(target)
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

impl Scorer for CustomTermScorer {
    fn score(&mut self) -> Score {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.score(fieldnorm_id, term_freq)
    }
}
