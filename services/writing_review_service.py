from workflows.writing_review_workflow import WritingReviewWorkflow, UserLetterRequest, WrittenExamEvaluation

class WritingReviewService:
    """Service to handle evaluation of writing exam responses."""

    def __init__(self):
        self.workflow = WritingReviewWorkflow()

    async def evaluate_written_exam(self, user_letter_request: UserLetterRequest) -> WrittenExamEvaluation:
        """Evaluates the user's written exam response and returns corrections."""
        evaluation: WrittenExamEvaluation = await self.workflow.evaluate_written_exam(user_letter_request)
        return evaluation 