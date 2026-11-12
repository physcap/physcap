# Task: Pack the grocery bag

## Environment
- A table surface containing four coloured cubes and one paper grocery bag.
- The cubes are the groceries to be packed.

## Objective
You are packing the groceries into the paper bag. Items must be packed one at a
time, in a sensible order. Decide which single item should go into the bag
FIRST, put that item in the bag, and stop.

## Operational Protocol
1. Observe & Anchor: Analyze the image. Identify all cubes and assign IDs based on colour and position (e.g., "Blue cube left").
2. Reason: Output the reasoning thought for the proposed action.
   - Output Format: THOUGHT: [Reasoning] | ACTION: [Action on cube ID].
3. Commit: Once you have decided which item goes in first, print exactly one line
   in this format, on its own line:
     FIRST ITEM: [cube ID]
   Then place that cube in the paper bag.

## Constraints
- Sequential Interaction: Interact with exactly one cube per reasoning cycle.
- Single Placement: Put exactly one cube in the bag. Leave the other three on the table.
- Termination: End the task once the first item is in the paper bag.
