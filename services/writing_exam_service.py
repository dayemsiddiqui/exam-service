from workflows.writing_exam_workflow import WritingExamWorkflow, WritingExam

class WritingExamService:
    """Service to handle the generation of letter writing exams."""

    def __init__(self):
        self.workflow = WritingExamWorkflow()

    async def get_writing_exam(self) -> WritingExam:
        """Generates a letter writing exam containing a letter and four tasks."""
        exam: WritingExam = await self.workflow.generate_writing_exam()
        return exam 